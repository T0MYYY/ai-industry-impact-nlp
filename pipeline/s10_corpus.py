"""Analysis corpus: English, de-syndicated documents and their non-boilerplate AI blocks.

Input is the 02D clean AI corpus. Three corrections are applied before any modeling:
  * language ID, because the source `language` field is "en" for every row, including
    Portuguese, Galician and Kazakh pages;
  * near-duplicate clustering of documents (MinHash LSH), so a syndicated wire story
    counts once;
  * block-level boilerplate flags for paragraphs repeated across many documents.
Press-release sources are flagged, not dropped, so results can be checked with and
without them.
"""

from __future__ import annotations

import os
import re
import subprocess
import urllib.request
from multiprocessing import Pool

import numpy as np
import pandas as pd

from . import config
from .common import normalize_spaces, setup_logging, text_key, timed, write_json

log = setup_logging("s10_corpus")

MIN_BLOCK_CHARS = 80
MIN_DOC_CHARS = 300
LANG_MIN_PROB = 0.70
MINHASH_PERM = 128
MINHASH_THRESHOLD = 0.8
SHINGLE = 5
BOILERPLATE_MIN_DOCS = 20

PRESS_WIRE_DOMAINS = {
    "prnewswire.com", "businesswire.com", "globenewswire.com", "einpresswire.com",
    "accesswire.com", "newswire.ca", "prweb.com", "openpr.com", "newsfilecorp.com",
    "prlog.org", "issuewire.com", "newswire.com", "prunderground.com", "24-7pressrelease.com",
    "menafn.com", "financialcontent.com", "prnewswire.co.uk", "accessnewswire.com", "newsdirect.com",
}
PRESS_URL_RE = re.compile(r"press[-_]?release|/news-release|/pressreleases?/", re.I)

LID_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"


def _lang_model():
    path = config.scratch("models", "lid.176.ftz")
    if not path.exists():
        urllib.request.urlretrieve(LID_URL, path)
    try:
        import fasttext  # provided by fasttext-predict
    except ImportError:
        subprocess.run(["pip", "install", "-q", "fasttext-predict"], check=True)
        import fasttext
    return fasttext.load_model(str(path))


def detect_language(texts: list[str]) -> tuple[list[str], np.ndarray]:
    model = _lang_model()
    langs, probs = [], np.zeros(len(texts), dtype=np.float32)
    for i, t in enumerate(texts):
        (label,), (p,) = model.predict(t.replace("\n", " ")[:1500], k=1)
        langs.append(label.replace("__label__", ""))
        probs[i] = p
    return langs, probs


_TOKEN_RE = re.compile(r"[0-9a-z]+")


def _minhash(text: str):
    from datasketch import MinHash

    toks = _TOKEN_RE.findall(text.lower())[:2000]
    m = MinHash(num_perm=MINHASH_PERM, seed=config.SEED)
    if len(toks) < SHINGLE:
        toks = toks + [""] * (SHINGLE - len(toks))
    m.update_batch([" ".join(toks[i:i + SHINGLE]).encode() for i in range(len(toks) - SHINGLE + 1)])
    return m


def near_duplicate_clusters(doc_ids: np.ndarray, texts: list[str]) -> np.ndarray:
    """Union-find over MinHash-LSH candidate pairs; returns a cluster id per document."""
    from datasketch import MinHashLSH

    with Pool(os.cpu_count()) as pool:
        sigs = pool.map(_minhash, texts, chunksize=500)
    lsh = MinHashLSH(threshold=MINHASH_THRESHOLD, num_perm=MINHASH_PERM)
    for i, m in enumerate(sigs):
        lsh.insert(str(i), m)

    parent = np.arange(len(texts))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, m in enumerate(sigs):
        for j in lsh.query(m):
            j = int(j)
            if j != i and m.jaccard(sigs[j]) >= MINHASH_THRESHOLD:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[max(ri, rj)] = min(ri, rj)
    roots = np.array([find(i) for i in range(len(texts))])
    return doc_ids[roots]


def is_press_release(domain: str, url: str) -> bool:
    d = (domain or "").lower()
    return any(d == w or d.endswith("." + w) for w in PRESS_WIRE_DOMAINS) or bool(PRESS_URL_RE.search(url or ""))


def main() -> None:
    subprocess.run(["pip", "install", "-q", "datasketch"], check=True)

    with timed(log, "load 02D outputs"):
        docs = pd.read_parquet(config.CLEAN_DOCS)
        blocks = pd.read_parquet(config.CLEAN_BLOCKS)
    log.info("input: %d docs, %d blocks", len(docs), len(blocks))

    docs["title"] = docs["title"].map(normalize_spaces)
    docs["text"] = docs["clean_ai_doc_text"].map(normalize_spaces)
    docs = docs.drop(columns=["clean_ai_doc_text"])
    docs["date"] = pd.to_datetime(docs["date"])

    with timed(log, "language id"):
        docs["lang"], docs["lang_prob"] = detect_language((docs["title"] + " " + docs["text"]).tolist())
    docs["is_english"] = (docs["lang"] == "en") & (docs["lang_prob"] >= LANG_MIN_PROB)

    with timed(log, "near-duplicate clustering"):
        docs = docs.sort_values(["date", "doc_id"]).reset_index(drop=True)
        docs["dup_cluster"] = near_duplicate_clusters(docs["doc_id"].to_numpy(), docs["text"].tolist())
    # Documents are sorted by date, so the cluster root is the earliest copy.
    docs["is_canonical"] = docs["doc_id"] == docs["dup_cluster"]
    docs["cluster_size"] = docs.groupby("dup_cluster")["doc_id"].transform("size")
    docs["is_press_release"] = [is_press_release(d, u) for d, u in zip(docs["domain"], docs["url"])]
    docs["text_len"] = docs["text"].str.len()
    docs["in_analysis"] = docs["is_english"] & docs["is_canonical"] & (docs["text_len"] >= MIN_DOC_CHARS)

    blocks["text"] = blocks["clean_block_text"].map(normalize_spaces)
    blocks = blocks.drop(columns=["clean_block_text"])
    blocks["text_len"] = blocks["text"].str.len()
    blocks["text_hash"] = blocks["text"].map(text_key)
    blocks = blocks.merge(docs[["doc_id", "in_analysis"]], on="doc_id", how="left")
    blocks["in_analysis"] = blocks["in_analysis"].fillna(False).astype(bool)
    n_docs_per_hash = blocks[blocks["in_analysis"]].groupby("text_hash")["doc_id"].nunique()
    blocks["hash_doc_count"] = blocks["text_hash"].map(n_docs_per_hash).fillna(0).astype(int)
    blocks["is_boilerplate"] = blocks["hash_doc_count"] >= BOILERPLATE_MIN_DOCS
    blocks["keep"] = blocks["in_analysis"] & ~blocks["is_boilerplate"] & (blocks["text_len"] >= MIN_BLOCK_CHARS)
    # Exact repeats below the boilerplate threshold keep their first copy only.
    kept = blocks.index[blocks["keep"]]
    repeat = blocks.loc[kept].duplicated("text_hash")
    blocks.loc[repeat.index[repeat], "keep"] = False

    n_keep = blocks[blocks["keep"]].groupby("doc_id").size()
    docs["n_blocks_kept"] = docs["doc_id"].map(n_keep).fillna(0).astype(int)
    docs.loc[docs["n_blocks_kept"] == 0, "in_analysis"] = False

    docs.to_parquet(config.out("corpus", "docs.parquet"), index=False)
    blocks.to_parquet(config.out("corpus", "blocks.parquet"), index=False)

    ana = docs[docs["in_analysis"]]
    stats = {
        "input_docs": len(docs),
        "input_blocks": len(blocks),
        "non_english_docs": int((~docs["is_english"]).sum()),
        "lang_counts_top": docs["lang"].value_counts().head(10).to_dict(),
        "near_dup_clusters_multi": int((docs.groupby("dup_cluster").size() > 1).sum()),
        "non_canonical_docs": int((~docs["is_canonical"]).sum()),
        "analysis_docs": len(ana),
        "analysis_blocks": int(blocks["keep"].sum()),
        "boilerplate_blocks": int(blocks["is_boilerplate"].sum()),
        "press_release_share_analysis": float(ana["is_press_release"].mean()),
        "top_domains_analysis": ana["domain"].value_counts().head(40).to_dict(),
        "largest_clusters": docs.groupby("dup_cluster").agg(n=("doc_id", "size"), title=("title", "first"))
            .sort_values("n", ascending=False).head(15).to_dict("records"),
        "date_min": str(ana["date"].min().date()),
        "date_max": str(ana["date"].max().date()),
    }
    write_json(stats, config.out("corpus", "corpus_stats.json"))
    log.info("stats: %s", {k: v for k, v in stats.items() if not isinstance(v, (dict, list))})


if __name__ == "__main__":
    main()
