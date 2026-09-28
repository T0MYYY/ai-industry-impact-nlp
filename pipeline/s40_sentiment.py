"""Entity-conditioned AI-impact sentiment with the 06B roberta-large classifier.

Every (entity, block) pair is scored, up to a random sample of MAX_PAIRS_PER_ENTITY pairs
per entity; the sampling weight is kept so cross-entity aggregates stay unbiased. The
context is a window around the mention rather than the block prefix, so mentions late in
long blocks are not cut off.

Evaluation: a fresh test set of pairs never used in 06A/06B is labeled by the LLM with
the 06A prompt. A slice of the 06B validation set is relabeled with the same prompt to
measure agreement between the deepseek-chat labels used for training and deepseek-flash.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from . import config, llm
from .common import normalize_spaces, setup_logging, timed, truncate, write_json

log = setup_logging("s40_sentiment")

LABELS = ["positive", "negative", "mixed_or_unclear"]
MODEL_TYPES = ["company", "technology", "government_institution", "person"]
MIN_ENTITY_DOCS = 10
# The study subject itself; "impact of AI on AI" is not a meaningful target.
EXCLUDE_ENTITIES = {"artificial intelligence"}
MAX_PAIRS_PER_ENTITY = 3000
MAX_CONTEXT_CHARS = 1400
LEAD_CHARS = 500
MAX_LENGTH = 384
TEST_PER_TYPE = 375
DRIFT_N = 300

# Verbatim from 05A so labels are comparable with the training data.
SENTIMENT_SYSTEM_PROMPT = """
Return exactly one JSON object and nothing else.

Task:
Evaluate the AI-related impact sentiment toward the specified entity in the given local context.

Label meaning:
- positive: AI is beneficial, enabling, improving, accelerating, expanding, or strategically favorable for the entity.
- negative: AI is harmful, threatening, restrictive, damaging, costly, displacing, risky, or strategically unfavorable for the entity.
- mixed_or_unclear: impact is mixed, ambiguous, descriptive, neutral, or cannot be assigned confidently.

Rules:
- Judge AI impact on the entity, not general article tone.
- Use only the given local context.
- If context is descriptive without clear directional effect, use mixed_or_unclear.
- Prefer conservative labeling.

Output schema:
{
  "sentiment_label": "positive",
  "confidence": 0.88,
  "reason_short": "Short explanation, <= 24 words."
}
""".strip()


def context_window(text: str, start: int) -> str:
    text = str(text)
    if len(text) <= MAX_CONTEXT_CHARS:
        return text
    s = max(0, min(start - LEAD_CHARS, len(text) - MAX_CONTEXT_CHARS))
    if s > 0:
        sp = text.find(" ", s)
        s = sp + 1 if 0 <= sp < s + 40 else s
    out = text[s:s + MAX_CONTEXT_CHARS].rstrip()
    return ("... " if s > 0 else "") + out + (" ..." if s + MAX_CONTEXT_CHARS < len(text) else "")


def model_input(entity: str, etype: str, title: str, domain: str, context: str) -> str:
    # Same layout as build_model_input in 05B.
    parts = [f"Target entity: {entity}", f"Entity type: {etype}"]
    if title:
        parts.append(f"Document title: {title}")
    if domain:
        parts.append(f"Source domain: {domain}")
    parts.append("Instruction: classify the sentiment of AI-related impact on the target entity "
                 "in this context as positive, negative, or mixed_or_unclear.")
    parts.append(f"Context: {context}")
    return "\n".join(parts)


def build_pairs() -> pd.DataFrame:
    ents = pd.read_parquet(config.out("entities", "entities.parquet"))
    ents = ents[ents["type"].isin(MODEL_TYPES) & (ents["n_docs"] >= MIN_ENTITY_DOCS)
                & ~ents["entity_id"].isin(EXCLUDE_ENTITIES)]
    m = pd.read_parquet(config.out("entities", "mentions.parquet"),
                        columns=["entity_id", "doc_id", "block_id", "block_uid", "date", "domain", "char_start"])
    m = m[m["entity_id"].isin(set(ents["entity_id"]))]
    pairs = m.sort_values("char_start").drop_duplicates(["entity_id", "block_uid"])
    n_total = pairs.groupby("entity_id").size()
    pairs = (pairs.groupby("entity_id", group_keys=False)
             .apply(lambda g: g.sample(n=min(len(g), MAX_PAIRS_PER_ENTITY), random_state=config.SEED)))
    pairs["sample_weight"] = pairs["entity_id"].map(n_total) / pairs["entity_id"].map(pairs.groupby("entity_id").size())
    pairs = pairs.merge(ents[["entity_id", "name", "type"]], on="entity_id")

    blocks = pd.read_parquet(config.out("corpus", "blocks.parquet"), columns=["block_uid", "text"])
    blocks = blocks[blocks["block_uid"].isin(set(pairs["block_uid"]))]
    docs = pd.read_parquet(config.out("corpus", "docs.parquet"), columns=["doc_id", "title", "is_press_release"])
    pairs = pairs.merge(blocks, on="block_uid").merge(docs, on="doc_id")
    pairs["context"] = [context_window(t, s) for t, s in zip(pairs["text"], pairs["char_start"])]
    pairs = pairs.drop(columns=["text"]).reset_index(drop=True)
    log.info("pairs: %d over %d entities (%d before per-entity cap)", len(pairs), pairs["entity_id"].nunique(),
             int(n_total.sum()))
    return pairs


def predict(pairs: pd.DataFrame) -> np.ndarray:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(config.SENTIMENT_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(config.SENTIMENT_MODEL, torch_dtype=torch.bfloat16)
    model.cuda().eval()
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    order = [next(i for i, l in id2label.items() if l == lab) for lab in LABELS]

    texts = [model_input(n, t, normalize_spaces(ti), d, c) for n, t, ti, d, c in
             zip(pairs["name"], pairs["type"], pairs["title"], pairs["domain"], pairs["context"])]
    lengths = np.array([len(t) for t in texts])
    idx = np.argsort(lengths)
    probs = np.zeros((len(texts), 3), dtype=np.float32)
    bs = 256
    with torch.inference_mode():
        for i in range(0, len(idx), bs):
            b = idx[i:i + bs]
            enc = tok([texts[j] for j in b], truncation=True, max_length=MAX_LENGTH, padding=True, return_tensors="pt")
            logits = model(**{k: v.cuda() for k, v in enc.items()}).logits.float()
            probs[b] = torch.softmax(logits, -1)[:, order].cpu().numpy()
            if (i // bs) % 500 == 0:
                log.info("predicted %d / %d", i + len(b), len(texts))
    return probs


def validate_sentiment(d: dict) -> dict:
    lab = d.get("sentiment_label")
    if lab not in LABELS:
        raise ValueError(f"bad label {lab}")
    return {"sentiment_label": lab, "confidence": float(d.get("confidence", 0) or 0)}


def llm_payload(entity, etype, date, domain, title, context, n_docs) -> str:
    return json.dumps({
        "entity": entity, "entity_type": etype,
        "date": pd.Timestamp(date).date().isoformat() if pd.notna(date) else None,
        "domain": domain, "title": truncate(title, 180), "context": truncate(context, MAX_CONTEXT_CHARS),
        "entity_stats": {"n_docs": int(n_docs)},
    }, ensure_ascii=False, separators=(",", ":"))


def evaluation(pairs: pd.DataFrame, probs: np.ndarray, limit: int | None) -> dict:
    from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix, f1_score

    train = pd.read_parquet(config.SENTIMENT_LABELS, columns=["doc_id", "block_id"])
    used = set(zip(train["doc_id"], train["block_id"]))
    ents = pd.read_parquet(config.out("entities", "entities.parquet"), columns=["entity_id", "n_docs"])
    cand = pairs.assign(row=np.arange(len(pairs))).merge(ents, on="entity_id")
    cand = cand[[(d, b) not in used for d, b in zip(cand["doc_id"], cand["block_id"])]]
    # One pair per entity keeps the test set from being dominated by the largest entities.
    cand = cand.sample(frac=1.0, random_state=config.SEED).drop_duplicates("entity_id")
    test = cand.groupby("type", group_keys=False).apply(lambda g: g.head(TEST_PER_TYPE))

    jobs = [llm.Job(id=f"test_{r.entity_id}||{r.block_uid}", system=SENTIMENT_SYSTEM_PROMPT,
                    user=llm_payload(r.name, r.type, r.date, r.domain, r.title, r.context, r.n_docs),
                    meta={"row": int(r.row)}) for r in test.itertuples()]
    valid = pd.read_parquet(config.SENTIMENT_VALID).sample(n=DRIFT_N, random_state=config.SEED)
    for r in valid.itertuples():
        jobs.append(llm.Job(id=f"drift_{r.sample_id}", system=SENTIMENT_SYSTEM_PROMPT,
                            user=llm_payload(r.canonical_entity, r.final_type, r.date, r.domain, r.title,
                                             r.context_text, r.n_docs),
                            meta={"orig_label": r.sentiment_label}))
    path = config.out("sentiment", "sentiment_test_llm.jsonl")
    llm.run_batch("sentiment_test", jobs, path, validate_sentiment, max_tokens=90, limit=limit)
    res = llm.read_results(path)

    t = [(r["meta"]["row"], r["label"]["sentiment_label"]) for r in res if r["id"].startswith("test_")]
    rows = np.array([x[0] for x in t])
    y_true = [x[1] for x in t]
    y_pred = [LABELS[i] for i in probs[rows].argmax(1)] if len(rows) else []
    types = pairs["type"].to_numpy()[rows] if len(rows) else []
    out = {"n_test": len(t)}
    if t:
        out.update({
            "macro_f1": f1_score(y_true, y_pred, labels=LABELS, average="macro"),
            "report": classification_report(y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0),
            "confusion": {"labels": LABELS, "matrix": confusion_matrix(y_true, y_pred, labels=LABELS).tolist()},
            "macro_f1_by_type": {ty: f1_score([a for a, k in zip(y_true, types) if k == ty],
                                              [b for b, k in zip(y_pred, types) if k == ty],
                                              labels=LABELS, average="macro") for ty in MODEL_TYPES},
            "label_dist_test": pd.Series(y_true).value_counts(normalize=True).to_dict(),
        })
    d = [(r["meta"]["orig_label"], r["label"]["sentiment_label"]) for r in res if r["id"].startswith("drift_")]
    if d:
        a, b = zip(*d)
        out["labeler_drift"] = {"n": len(d), "agreement": float(np.mean(np.array(a) == np.array(b))),
                                "cohen_kappa": cohen_kappa_score(a, b)}
    return out


def main(limit: int | None = None, skip_predict: bool = False) -> None:
    pred_path = config.out("sentiment", "predictions.parquet")
    if skip_predict and pred_path.exists():
        pairs = pd.read_parquet(pred_path)
        probs = pairs[["p_positive", "p_negative", "p_mixed"]].to_numpy()
    else:
        with timed(log, "build pairs"):
            pairs = build_pairs()
        with timed(log, f"predict {len(pairs)} pairs"):
            probs = predict(pairs)
        pairs["p_positive"], pairs["p_negative"], pairs["p_mixed"] = probs[:, 0], probs[:, 1], probs[:, 2]
        pairs["pred"] = np.array(LABELS)[probs.argmax(1)]
        pairs.to_parquet(pred_path, index=False)

    metrics = evaluation(pairs, probs, limit)
    write_json(metrics, config.out("sentiment", "test_metrics.json"))
    log.info("test: %s", {k: v for k, v in metrics.items() if k in ("n_test", "macro_f1", "labeler_drift")})


if __name__ == "__main__":
    lim = next((int(a.split("=")[1]) for a in sys.argv if a.startswith("--limit=")), None)
    main(limit=lim, skip_predict="--skip-predict" in sys.argv)
