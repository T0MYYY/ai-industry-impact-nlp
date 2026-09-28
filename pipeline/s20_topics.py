"""BERTopic over the full analysis block set, with GPU UMAP/HDBSCAN.

Block embeddings are saved and reused by s50 (document industry classifier), so the
embedding model is chosen once here: all-mpnet-base-v2 (the corpus is English-only
after s10).
"""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pandas as pd

from . import config, llm
from .common import setup_logging, timed, truncate, write_json
from .taxonomy import INDUSTRIES, NO_INDUSTRY, industry_menu

log = setup_logging("s20_topics")

EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"
EMBED_CHARS = 1500
UMAP_PARAMS = dict(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine")
HDBSCAN_PARAMS = dict(min_cluster_size=400, min_samples=40, cluster_selection_method="eom")
MAX_TOPICS = 80
OUTLIER_THRESHOLD = 0.35

# "ai" and its variants appear in nearly every block after AI filtering and would lead
# every topic's keyword list.
EXTRA_STOPWORDS = [
    "ai", "artificial", "intelligence", "artificial intelligence", "generative", "genai",
    "said", "says", "new", "use", "using", "used", "also", "like", "just", "year", "years",
    "company", "companies", "technology", "technologies", "percent", "including",
]

EMB_PATH = config.out("topics", "block_embeddings_f16.npy")
EMB_INDEX_PATH = config.out("topics", "block_embeddings_index.parquet")


def ensure_packages() -> None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "bertopic"], check=True)


def embed(texts: list[str]) -> np.ndarray:
    import torch
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBED_MODEL, device="cuda")
    model.half()
    emb = model.encode(texts, batch_size=512, show_progress_bar=False, convert_to_numpy=True,
                       normalize_embeddings=True)
    del model
    torch.cuda.empty_cache()
    return emb.astype(np.float32)


def fit_topics(texts: list[str], emb: np.ndarray):
    from bertopic import BERTopic
    from bertopic.representation import KeyBERTInspired
    from cuml.cluster import HDBSCAN
    from cuml.manifold import UMAP
    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer

    vectorizer = CountVectorizer(
        stop_words=list(ENGLISH_STOP_WORDS.union(EXTRA_STOPWORDS)),
        ngram_range=(1, 2), min_df=20, max_df=0.5,
    )
    model = BERTopic(
        embedding_model=SentenceTransformer(EMBED_MODEL, device="cuda"),
        umap_model=UMAP(**UMAP_PARAMS, random_state=config.SEED),
        hdbscan_model=HDBSCAN(**HDBSCAN_PARAMS, prediction_data=True),
        vectorizer_model=vectorizer,
        representation_model=KeyBERTInspired(),
        calculate_probabilities=False,
        verbose=True,
    )
    topics, _ = model.fit_transform(texts, embeddings=emb)
    n_topics = len(set(topics)) - (1 if -1 in topics else 0)
    log.info("HDBSCAN: %d topics, outlier share %.3f", n_topics, np.mean(np.array(topics) == -1))
    if n_topics > MAX_TOPICS:
        model.reduce_topics(texts, nr_topics=MAX_TOPICS)
        topics = model.topics_
    outlier_share_raw = float(np.mean(np.array(topics) == -1))
    new_topics = model.reduce_outliers(texts, topics, strategy="embeddings", embeddings=emb,
                                       threshold=OUTLIER_THRESHOLD)
    model.update_topics(texts, topics=new_topics, vectorizer_model=vectorizer,
                        representation_model=KeyBERTInspired())
    return model, np.array(new_topics), outlier_share_raw


TOPIC_SYSTEM = f"""You label topic clusters from a news corpus about artificial intelligence.
Given the keywords and sample passages of one cluster, return a JSON object:
{{"label": "<=5 word descriptive name", "description": "<=25 words", "industry": "<one key>", "is_coherent": true|false}}
"industry" must be one of these keys (use "{NO_INDUSTRY}" when the cluster is about AI in general rather than a sector):
{industry_menu()}
Set is_coherent false if the passages do not share a clear subject (e.g. mixed boilerplate, price tickers, navigation text)."""


def validate_topic(d: dict) -> dict:
    ind = d.get("industry")
    if ind not in INDUSTRIES and ind != NO_INDUSTRY:
        raise ValueError(f"bad industry {ind}")
    return {"label": str(d["label"])[:60], "description": str(d.get("description", ""))[:240],
            "industry": ind, "is_coherent": bool(d.get("is_coherent", True))}


def label_topics(info: pd.DataFrame, reps: dict[int, list[str]], limit: int | None = None) -> pd.DataFrame:
    jobs = []
    for _, r in info[info["Topic"] >= 0].iterrows():
        tid = int(r["Topic"])
        kw = ", ".join(w for w, _ in r["Representation_pairs"][:12])
        passages = "\n".join(f"{i + 1}. {truncate(t, 400)}" for i, t in enumerate(reps[tid][:6]))
        jobs.append(llm.Job(id=f"topic_{tid}", system=TOPIC_SYSTEM,
                            user=f"Keywords: {kw}\n\nPassages:\n{passages}", meta={"topic": tid}))
    path = config.out("topics", "topic_labels_llm.jsonl")
    llm.run_batch("topic_labels", jobs, path, validate_topic, max_tokens=160, concurrency=40, limit=limit)
    rows = [{"topic": r["meta"]["topic"], **r["label"]} for r in llm.read_results(path)]
    return pd.DataFrame(rows)


def main(label_only: bool = False, fit_only: bool = False) -> None:
    ensure_packages()
    blocks = pd.read_parquet(config.out("corpus", "blocks.parquet"),
                             columns=["block_uid", "doc_id", "block_id", "date", "domain", "text", "keep"])
    blocks = blocks[blocks["keep"]].drop(columns="keep").reset_index(drop=True)
    texts = [t[:EMBED_CHARS] for t in blocks["text"]]
    log.info("blocks for topic model: %d", len(blocks))

    if EMB_PATH.exists():
        idx = pd.read_parquet(EMB_INDEX_PATH)
        assert idx["block_uid"].tolist() == blocks["block_uid"].tolist(), "embedding index out of date"
        emb = np.load(EMB_PATH).astype(np.float32)
    else:
        with timed(log, f"embed {len(texts)} blocks"):
            emb = embed(texts)
        np.save(EMB_PATH, emb.astype(np.float16))
        blocks[["block_uid", "doc_id"]].to_parquet(EMB_INDEX_PATH, index=False)

    model_dir = config.out("topics", "bertopic_model", "x").parent
    assign_path = config.out("topics", "block_topics.parquet")
    if not label_only:
        with timed(log, "fit BERTopic"):
            model, topics, outlier_raw = fit_topics(texts, emb)
        model.save(str(model_dir), serialization="safetensors", save_ctfidf=True, save_embedding_model=EMBED_MODEL)
        blocks["topic"] = topics
        blocks[["block_uid", "doc_id", "block_id", "date", "domain", "topic"]].to_parquet(assign_path, index=False)

        info = model.get_topic_info()
        info["Representation_pairs"] = [model.get_topic(t) if t >= 0 else [] for t in info["Topic"]]
        reps = model.get_representative_docs()
        info.to_pickle(config.out("topics", "topic_info.pkl"))
        with open(config.out("topics", "topic_reps.json"), "w") as f:
            json.dump({int(k): v for k, v in reps.items()}, f)
        write_json({
            "n_blocks": len(blocks), "n_topics": int((info["Topic"] >= 0).sum()),
            "outlier_share_before_reduction": outlier_raw,
            "outlier_share_after_reduction": float(np.mean(topics == -1)),
            "umap": UMAP_PARAMS, "hdbscan": HDBSCAN_PARAMS, "embed_model": EMBED_MODEL,
            "outlier_threshold": OUTLIER_THRESHOLD,
        }, config.out("topics", "topic_model_stats.json"))
    else:
        info = pd.read_pickle(config.out("topics", "topic_info.pkl"))
        with open(config.out("topics", "topic_reps.json")) as f:
            reps = {int(k): v for k, v in json.load(f).items()}

    if fit_only:
        return
    labels = label_topics(info, reps)
    summary = info[["Topic", "Count", "Name"]].rename(columns={"Topic": "topic", "Count": "n_blocks", "Name": "bertopic_name"})
    summary["keywords"] = [", ".join(w for w, _ in p[:10]) for p in info["Representation_pairs"]]
    summary = summary.merge(labels, on="topic", how="left")

    assign = pd.read_parquet(assign_path)
    n_docs = assign[assign["topic"] >= 0].groupby("topic")["doc_id"].nunique()
    summary["n_docs"] = summary["topic"].map(n_docs).fillna(0).astype(int)
    summary.to_csv(config.out("topics", "topic_summary.csv"), index=False)

    # Document-level topic shares by block count; the dominant topic ignores outliers.
    dt = assign[assign["topic"] >= 0].groupby(["doc_id", "topic"]).size().rename("n").reset_index()
    dt["share"] = dt["n"] / dt.groupby("doc_id")["n"].transform("sum")
    dt.to_parquet(config.out("topics", "doc_topic_shares.parquet"), index=False)
    dom = dt.sort_values(["doc_id", "n", "topic"], ascending=[True, False, True]).drop_duplicates("doc_id")
    dom[["doc_id", "topic", "share"]].rename(columns={"topic": "dominant_topic", "share": "dominant_share"}) \
        .to_parquet(config.out("topics", "doc_dominant_topic.parquet"), index=False)
    log.info("topics labeled: %d; docs with a topic: %d", labels.shape[0], dom.shape[0])


if __name__ == "__main__":
    main(label_only="--label-only" in sys.argv, fit_only="--fit-only" in sys.argv)
