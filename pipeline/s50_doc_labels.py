"""Document-level industry, impact direction, mechanism, adoption outcome and factors.

A quarter-stratified random sample of analysis documents is labeled by the LLM with one
structured prompt. The sample is representative of the corpus, so mechanism, outcome and
factor shares are estimated on it directly (with bootstrap intervals in s60). Industry
labels are extended to every analysis document with one-vs-rest logistic regressions on
mean block embeddings from s20, validated on a held-out part of the sample.

The prompt does not include any model sentiment scores, so industry assignment and the
s40 sentiment estimates stay independent.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from . import config, llm
from .common import setup_logging, timed, truncate, write_json
from .taxonomy import (DIRECTIONS, FACTOR_ROLES, FACTORS, INDUSTRIES, MECHANISMS, NO_INDUSTRY, OUTCOMES,
                       industry_menu, menu)

log = setup_logging("s50_doc_labels")

SAMPLE_N = 15000
DOC_CHARS = 2500

DOC_SYSTEM = f"""You annotate news articles about artificial intelligence for an industry-impact study.
Return one JSON object:
{{"industries": [<up to 2 industry keys, most central first>],
 "direction": "<direction>",
 "mechanisms": [<0-3 mechanism keys>],
 "outcome": "<outcome key>",
 "factors": [{{"f": "<factor key>", "r": "enabler|barrier"}}, ...up to 4]}}

industries: sectors whose business or work the article says AI affects. Use [] if the article is about AI generally (models, funding, policy debate) without a specific affected sector.
{industry_menu()}

direction: net effect of AI on the first listed industry as the article describes it: {", ".join(DIRECTIONS)}. Use neutral when industries is [] or the article only reports facts.

mechanisms: how AI changes work in the industry, only if the article states it; [] otherwise:
{menu(MECHANISMS)}

outcome: adoption status described:
{menu(OUTCOMES)}

factors: conditions the article explicitly says help (enabler) or hinder (barrier) adopting AI:
{menu(FACTORS)}

Most articles state no mechanism and no factor; leave those lists empty rather than guess.
Price pages for crypto tokens, stock tips and product listicles usually have industries [] or only the vendor's sector, outcome none or product_launch, and no factors.
Use only keys from these lists. Judge from the article text, not from the outlet."""


def validate_doc(d: dict) -> dict:
    inds = [i for i in (d.get("industries") or []) if i in INDUSTRIES][:2]
    direction = d.get("direction")
    if direction not in DIRECTIONS:
        raise ValueError(f"bad direction {direction}")
    outcome = d.get("outcome")
    if outcome not in OUTCOMES:
        raise ValueError(f"bad outcome {outcome}")
    mechs = sorted({m for m in (d.get("mechanisms") or []) if m in MECHANISMS})
    facs = []
    for f in (d.get("factors") or [])[:4]:
        if isinstance(f, dict) and f.get("f") in FACTORS and f.get("r") in FACTOR_ROLES:
            facs.append({"f": f["f"], "r": f["r"]})
    return {"industries": inds, "direction": direction, "mechanisms": mechs, "outcome": outcome, "factors": facs}


def draw_sample(docs: pd.DataFrame) -> pd.DataFrame:
    ana = docs[docs["in_analysis"]].copy()
    ana["quarter"] = ana["date"].dt.to_period("Q").astype(str)
    frac = SAMPLE_N / len(ana)
    s = ana.groupby("quarter", group_keys=False)[ana.columns.tolist()].apply(
        lambda g: g.sample(n=max(1, int(round(len(g) * frac))), random_state=config.SEED))
    return s.sort_values("doc_id").reset_index(drop=True)


def flatten(results: list[dict]) -> pd.DataFrame:
    rows = []
    for r in results:
        lab = r["label"]
        row = {"doc_id": int(r["id"]), "direction": lab["direction"], "outcome": lab["outcome"],
               "industry_primary": lab["industries"][0] if lab["industries"] else NO_INDUSTRY,
               "industries": "|".join(lab["industries"]), "mechanisms": "|".join(lab["mechanisms"])}
        for k in INDUSTRIES:
            row[f"ind_{k}"] = int(k in lab["industries"])
        for k in MECHANISMS:
            row[f"mech_{k}"] = int(k in lab["mechanisms"])
        for k in FACTORS:
            roles = {f["r"] for f in lab["factors"] if f["f"] == k}
            row[f"fac_{k}_enabler"] = int("enabler" in roles)
            row[f"fac_{k}_barrier"] = int("barrier" in roles)
        rows.append(row)
    return pd.DataFrame(rows)


def doc_embeddings(doc_ids: np.ndarray) -> np.ndarray:
    idx = pd.read_parquet(config.out("topics", "block_embeddings_index.parquet"))
    emb = np.load(config.out("topics", "block_embeddings_f16.npy"), mmap_mode="r")
    pos = pd.Series(np.arange(len(idx)), index=idx["doc_id"].to_numpy())
    out = np.zeros((len(doc_ids), emb.shape[1]), dtype=np.float32)
    grouped = pos.groupby(level=0).apply(lambda s: s.to_numpy())
    for i, d in enumerate(doc_ids):
        rows = grouped.get(d)
        if rows is not None:
            v = np.asarray(emb[rows], dtype=np.float32).mean(0)
            out[i] = v / (np.linalg.norm(v) + 1e-9)
    return out


def train_industry_classifier(sample: pd.DataFrame, docs: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
    from sklearn.model_selection import train_test_split

    X = doc_embeddings(sample["doc_id"].to_numpy())
    tr, te = train_test_split(np.arange(len(sample)), test_size=0.3, random_state=config.SEED)
    va, te = te[: len(te) // 2], te[len(te) // 2:]
    all_ids = docs.loc[docs["in_analysis"], "doc_id"].to_numpy()
    with timed(log, "embed all analysis docs"):
        X_all = doc_embeddings(all_ids)

    metrics, preds = {}, {"doc_id": all_ids}
    for k in INDUSTRIES:
        y = sample[f"ind_{k}"].to_numpy()
        if y[tr].sum() < 20:
            metrics[k] = {"skipped": True, "n_pos": int(y.sum())}
            continue
        clf = LogisticRegression(C=2.0, max_iter=2000, class_weight="balanced")
        clf.fit(X[tr], y[tr])
        p_va = clf.predict_proba(X[va])[:, 1]
        grid = np.linspace(0.2, 0.95, 31)
        f1s = [f1_score(y[va], p_va >= t, zero_division=0) for t in grid]
        thr = float(grid[int(np.argmax(f1s))])
        p_te = clf.predict_proba(X[te])[:, 1]
        metrics[k] = {
            "n_pos": int(y.sum()), "threshold": thr,
            "test_precision": precision_score(y[te], p_te >= thr, zero_division=0),
            "test_recall": recall_score(y[te], p_te >= thr, zero_division=0),
            "test_f1": f1_score(y[te], p_te >= thr, zero_division=0),
            "test_ap": average_precision_score(y[te], p_te) if y[te].sum() else None,
            "test_prevalence_llm": float(y[te].mean()), "test_prevalence_pred": float((p_te >= thr).mean()),
        }
        p_all = clf.predict_proba(X_all)[:, 1]
        preds[f"p_{k}"] = p_all.astype(np.float32)
        preds[f"pred_{k}"] = (p_all >= thr).astype(np.int8)
    return pd.DataFrame(preds), metrics


def main(limit: int | None = None, label_only: bool = False) -> None:
    docs = pd.read_parquet(config.out("corpus", "docs.parquet"),
                           columns=["doc_id", "date", "title", "text", "domain", "in_analysis", "is_press_release"])
    sample_path = config.out("industry", "doc_sample.parquet")
    if sample_path.exists():
        sample = pd.read_parquet(sample_path)
    else:
        sample = draw_sample(docs)
        sample[["doc_id", "date", "quarter", "domain", "is_press_release"]].to_parquet(sample_path, index=False)
    text = docs.set_index("doc_id").loc[sample["doc_id"]]
    jobs = [llm.Job(id=str(d), system=DOC_SYSTEM, user=f"Title: {truncate(t, 200)}\n\nArticle:\n{truncate(x, DOC_CHARS)}")
            for d, t, x in zip(sample["doc_id"], text["title"], text["text"])]
    path = config.out("industry", "doc_labels_llm.jsonl")
    llm.run_batch("doc_labels", jobs, path, validate_doc, max_tokens=180, limit=limit)

    lab = flatten(llm.read_results(path))
    lab = sample[["doc_id", "date", "quarter", "domain", "is_press_release"]].merge(lab, on="doc_id")
    lab.to_parquet(config.out("industry", "doc_labels_sample.parquet"), index=False)
    log.info("labeled sample docs: %d / %d", len(lab), len(sample))
    if label_only or len(lab) < 0.9 * len(sample):
        return

    preds, metrics = train_industry_classifier(lab, docs)
    preds.to_parquet(config.out("industry", "doc_industry_pred.parquet"), index=False)
    write_json(metrics, config.out("industry", "industry_classifier_metrics.json"))
    log.info("classifier test F1: %s", {k: round(v.get("test_f1", 0), 3) for k, v in metrics.items()})


if __name__ == "__main__":
    lim = next((int(a.split("=")[1]) for a in sys.argv if a.startswith("--limit=")), None)
    main(limit=lim, label_only="--label-only" in sys.argv)
