"""Result tables for the three research questions.

All intervals are 95% percentile bootstrap intervals over documents (Poisson weights,
B resamples). Every number quoted in the deck and README is written to results.json
from here, so the deliverables read from one place.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import config
from .common import setup_logging, write_json
from .taxonomy import FACTORS, INDUSTRIES, MECHANISMS, NO_INDUSTRY, OUTCOMES, SUPPLY_SIDE

log = setup_logging("s60_analysis")

B = 1000
MIN_ENTITY_DOCS = 100
MIN_INDUSTRY_DOCS = 150
CHATGPT_LAUNCH = pd.Timestamp("2022-11-30")
RNG = np.random.default_rng(config.SEED)


def tables_path(name: str):
    return config.out("tables", name)


def boot_ratio(num: np.ndarray, den: np.ndarray | None = None) -> tuple[float, float, float]:
    """Estimate sum(num)/sum(den) (den defaults to ones) with a Poisson bootstrap over rows."""
    num = np.asarray(num, dtype=float)
    den = np.ones_like(num) if den is None else np.asarray(den, dtype=float)
    if den.sum() == 0:
        return np.nan, np.nan, np.nan
    est = num.sum() / den.sum()
    w = RNG.poisson(1.0, size=(B, len(num)))
    reps = (w @ num) / np.maximum(w @ den, 1e-12)
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return float(est), float(lo), float(hi)


def ci_row(prefix: str, t: tuple[float, float, float]) -> dict:
    return {prefix: t[0], f"{prefix}_lo": t[1], f"{prefix}_hi": t[2]}


# RQ1 ---------------------------------------------------------------------------------

def industry_exposure(lab: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for k, desc in INDUSTRIES.items():
        y = lab[f"ind_{k}"].to_numpy()
        prim = (lab["industry_primary"] == k).to_numpy()
        no_pr = ~lab["is_press_release"].to_numpy()
        rows.append({
            "industry": k, "label": desc, "supply_side": k in SUPPLY_SIDE, "n_docs_sample": int(y.sum()),
            **ci_row("share_any", boot_ratio(y)),
            **ci_row("share_primary", boot_ratio(prim)),
            "share_any_excl_press": float(y[no_pr].mean()),
        })
    none = (lab["industry_primary"] == NO_INDUSTRY).to_numpy()
    rows.append({"industry": NO_INDUSTRY, "label": "No specific industry", "supply_side": False,
                 "n_docs_sample": int(none.sum()), **ci_row("share_any", boot_ratio(none)),
                 **ci_row("share_primary", boot_ratio(none)), "share_any_excl_press": float(none[~lab["is_press_release"]].mean())})
    return pd.DataFrame(rows).sort_values("share_any", ascending=False)


def industry_trend(pred: pd.DataFrame, docs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = pred.merge(docs[["doc_id", "date"]], on="doc_id")
    d["month"] = d["date"].dt.to_period("M").dt.to_timestamp()
    last_full = d["month"].max() - pd.offsets.MonthBegin(1)
    d = d[d["month"] <= last_full]
    cols = [c for c in d.columns if c.startswith("pred_")]
    monthly = d.groupby("month")[cols].mean().rename(columns=lambda c: c[5:])
    monthly["n_docs"] = d.groupby("month").size()
    early = d[d["date"] < CHATGPT_LAUNCH]
    late = d[d["date"] >= d["date"].max() - pd.DateOffset(years=1)]
    change = []
    for c in cols:
        k = c[5:]
        e, l = early[c].to_numpy(), late[c].to_numpy()
        we = RNG.poisson(1.0, size=(B, len(e)))
        wl = RNG.poisson(1.0, size=(B, len(l)))
        reps = (wl @ l) / wl.sum(1) - (we @ e) / we.sum(1)
        change.append({"industry": k, "share_pre_chatgpt": e.mean(), "share_last_12m": l.mean(),
                       "change_pp": 100 * (l.mean() - e.mean()),
                       "change_pp_lo": 100 * np.percentile(reps, 2.5), "change_pp_hi": 100 * np.percentile(reps, 97.5)})
    return monthly.reset_index(), pd.DataFrame(change).sort_values("change_pp", ascending=False)


# RQ2 ---------------------------------------------------------------------------------

def direction_by_industry(lab: pd.DataFrame, pairs: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    doc_si = doc_sentiment(pairs)
    rows = []
    for k in INDUSTRIES:
        sub = lab[lab["industry_primary"] == k]
        if len(sub) < MIN_INDUSTRY_DOCS:
            continue
        r = {"industry": k, "n_docs_sample": len(sub)}
        for dname in ["positive", "negative", "mixed", "neutral"]:
            r.update(ci_row(f"share_{dname}", boot_ratio((sub["direction"] == dname).to_numpy())))
        net = (sub["direction"] == "positive").astype(float) - (sub["direction"] == "negative").astype(float)
        r.update(ci_row("net_direction", boot_ratio(net.to_numpy())))
        # Entity-level sentiment in the same industry, from the classifier-assigned full corpus.
        ind_docs = pred.loc[pred.get(f"pred_{k}", pd.Series(0, index=pred.index)) == 1, "doc_id"]
        s = doc_si[doc_si["doc_id"].isin(set(ind_docs))]
        r.update(ci_row("entity_sentiment_index", boot_ratio(s["si"].to_numpy())))
        r["n_docs_entity_sentiment"] = len(s)
        rows.append(r)
    return pd.DataFrame(rows).sort_values("net_direction", ascending=False)


def mechanism_by_industry(lab: pd.DataFrame) -> pd.DataFrame:
    rows = []
    groups = [(k, lab[lab["industry_primary"] == k]) for k in INDUSTRIES] + [("all_sector_docs", lab[lab["industry_primary"] != NO_INDUSTRY])]
    for k, sub in groups:
        if len(sub) < MIN_INDUSTRY_DOCS:
            continue
        r = {"industry": k, "n_docs_sample": len(sub)}
        for m in MECHANISMS:
            r.update(ci_row(f"share_{m}", boot_ratio(sub[f"mech_{m}"].to_numpy())))
        rows.append(r)
    return pd.DataFrame(rows)


def doc_sentiment(pairs: pd.DataFrame) -> pd.DataFrame:
    p = pairs.assign(si=pairs["p_positive"] - pairs["p_negative"])
    return p.groupby("doc_id").agg(si=("si", "mean"), date=("date", "first"),
                                   is_press_release=("is_press_release", "first")).reset_index()


def entity_rankings(pairs: pd.DataFrame, ents: pd.DataFrame) -> pd.DataFrame:
    p = pairs.assign(si=pairs["p_positive"] - pairs["p_negative"], neg=pairs["pred"] == "negative",
                     pos=pairs["pred"] == "positive")
    per_doc = p.groupby(["entity_id", "doc_id"]).agg(si=("si", "mean"), pos=("pos", "mean"), neg=("neg", "mean"),
                                                    pr=("is_press_release", "first")).reset_index()
    rows = []
    for eid, g in per_doc.groupby("entity_id"):
        if len(g) < MIN_ENTITY_DOCS:
            continue
        si = boot_ratio(g["si"].to_numpy())
        rows.append({"entity_id": eid, "n_docs_scored": len(g), **ci_row("sentiment_index", si),
                     "share_positive": g["pos"].mean(), "share_negative": g["neg"].mean(),
                     "sentiment_index_excl_press": g.loc[~g["pr"], "si"].mean() if (~g["pr"]).sum() >= 30 else np.nan})
    out = pd.DataFrame(rows).merge(ents[["entity_id", "name", "type", "kind", "n_docs"]], on="entity_id")
    return out.sort_values("n_docs", ascending=False)


def sentiment_trend(pairs: pd.DataFrame) -> pd.DataFrame:
    d = doc_sentiment(pairs)
    d["month"] = pd.to_datetime(d["date"]).dt.to_period("M").dt.to_timestamp()
    last_full = d["month"].max() - pd.offsets.MonthBegin(1)
    d = d[d["month"] <= last_full]
    rows = []
    for mth, g in d.groupby("month"):
        rows.append({"month": mth, "n_docs": len(g), **ci_row("sentiment_index", boot_ratio(g["si"].to_numpy())),
                     "sentiment_index_excl_press": g.loc[~g["is_press_release"], "si"].mean()})
    return pd.DataFrame(rows)


# RQ3 ---------------------------------------------------------------------------------

def factor_table(lab: pd.DataFrame) -> pd.DataFrame:
    rows = []
    adopt = lab[lab["outcome"].isin(["deployed", "setback"])]
    setback = (adopt["outcome"] == "setback").to_numpy()
    base = setback.mean()
    for f in FACTORS:
        en, ba = lab[f"fac_{f}_enabler"].to_numpy(), lab[f"fac_{f}_barrier"].to_numpy()
        r = {"factor": f, "description": FACTORS[f], **ci_row("share_enabler", boot_ratio(en)),
             **ci_row("share_barrier", boot_ratio(ba))}
        # Setback rate among adoption stories that cite the factor as a barrier vs. not.
        cited = adopt[f"fac_{f}_barrier"].to_numpy() == 1
        r["n_adoption_docs_citing_barrier"] = int(cited.sum())
        if cited.sum() >= 30:
            r.update(ci_row("setback_rate_when_barrier", boot_ratio(setback[cited])))
            r["setback_rate_otherwise"] = float(setback[~cited].mean())
        cited_en = adopt[f"fac_{f}_enabler"].to_numpy() == 1
        if cited_en.sum() >= 30:
            r.update(ci_row("setback_rate_when_enabler", boot_ratio(setback[cited_en])))
        r["setback_rate_all_adoption"] = float(base)
        rows.append(r)
    return pd.DataFrame(rows).sort_values("share_barrier", ascending=False)


def factor_by_industry(lab: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for k in INDUSTRIES:
        sub = lab[lab["industry_primary"] == k]
        if len(sub) < MIN_INDUSTRY_DOCS:
            continue
        for f in FACTORS:
            rows.append({"industry": k, "factor": f, "n_docs_sample": len(sub),
                         "share_enabler": sub[f"fac_{f}_enabler"].mean(), "share_barrier": sub[f"fac_{f}_barrier"].mean()})
    return pd.DataFrame(rows)


def outcome_by_industry(lab: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for k in list(INDUSTRIES) + [NO_INDUSTRY]:
        sub = lab[lab["industry_primary"] == k]
        if len(sub) < MIN_INDUSTRY_DOCS:
            continue
        r = {"industry": k, "n_docs_sample": len(sub)}
        for o in OUTCOMES:
            r.update(ci_row(f"share_{o}", boot_ratio((sub["outcome"] == o).to_numpy())))
        rows.append(r)
    return pd.DataFrame(rows)


# Topics, entities and sentiment over time ------------------------------------------------

MIN_TOPIC_DOCS = 200
TREND_COMPANIES = ["openai", "google", "microsoft", "nvidia", "meta", "anthropic"]


def topic_quarterly_share(doc_topics: pd.DataFrame, docs: pd.DataFrame, topics: pd.DataFrame) -> pd.DataFrame:
    """Share of each quarter's articles that contain at least one block in the topic."""
    d = docs[docs["in_analysis"]][["doc_id", "date"]].copy()
    d["quarter"] = d["date"].dt.to_period("Q").astype(str)
    d = d[d["date"] < d["date"].max().to_period("Q").start_time]  # drop the partial last quarter
    n_q = d.groupby("quarter").size()
    t = doc_topics[["doc_id", "topic"]].merge(d[["doc_id", "quarter"]], on="doc_id")
    share = t.groupby(["topic", "quarter"]).size().div(n_q, level="quarter").rename("share")
    out = share.reset_index().merge(topics[["topic", "label", "industry"]], on="topic", how="left")
    return out


def topic_sentiment(pairs: pd.DataFrame, block_topics: pd.DataFrame, topics: pd.DataFrame) -> pd.DataFrame:
    """Entity sentiment of the pairs whose block belongs to each topic, averaged per article."""
    p = pairs.merge(block_topics[["block_uid", "topic"]], on="block_uid")
    p = p[p["topic"] >= 0].assign(si=lambda x: x["p_positive"] - x["p_negative"], neg=lambda x: x["pred"] == "negative")
    per_doc = p.groupby(["topic", "doc_id"]).agg(si=("si", "mean"), neg=("neg", "mean")).reset_index()
    rows = []
    for tid, g in per_doc.groupby("topic"):
        if len(g) < MIN_TOPIC_DOCS:
            continue
        rows.append({"topic": tid, "n_docs_scored": len(g), **ci_row("sentiment_index", boot_ratio(g["si"].to_numpy())),
                     "share_negative": g["neg"].mean()})
    out = pd.DataFrame(rows).merge(topics[["topic", "label", "industry", "n_docs", "is_coherent"]], on="topic")
    return out.sort_values("sentiment_index")


def entity_coverage(ents: pd.DataFrame, n: int = 15) -> pd.DataFrame:
    named = ents[(ents["kind"] == "named") & ents["type"].isin(["company", "technology", "government_institution", "person"])]
    return (named.sort_values("n_docs", ascending=False).groupby("type").head(n)
            [["type", "name", "n_docs", "n_mentions", "n_domains", "n_keys_merged"]])


def quarterly_sentiment(pairs: pd.DataFrame, key: pd.Series, groups: list) -> pd.DataFrame:
    """Quarterly article-level sentiment index for each group; `key` maps doc_id (or row) to group."""
    rows = []
    for gname, sub in pairs.groupby(key):
        if gname not in groups:
            continue
        d = sub.assign(si=sub["p_positive"] - sub["p_negative"]).groupby("doc_id").agg(si=("si", "mean"), date=("date", "first"))
        d["quarter"] = pd.to_datetime(d["date"]).dt.to_period("Q")
        d = d[d["quarter"] < pd.to_datetime(pairs["date"]).max().to_period("Q")]
        for q, g in d.groupby("quarter"):
            if len(g) >= 30:
                rows.append({"group": gname, "quarter": str(q), "n_docs": len(g),
                             **ci_row("sentiment_index", boot_ratio(g["si"].to_numpy()))})
    return pd.DataFrame(rows)


def industry_quarterly_sentiment(pairs: pd.DataFrame, pred: pd.DataFrame, industries: list) -> pd.DataFrame:
    frames = []
    for k in industries:
        ids = set(pred.loc[pred[f"pred_{k}"] == 1, "doc_id"])
        sub = pairs[pairs["doc_id"].isin(ids)].assign(industry=k)
        frames.append(quarterly_sentiment(sub, sub["industry"], [k]))
    return pd.concat(frames, ignore_index=True)


def _load(path):
    with open(path) as f:
        return json.load(f)


def main() -> None:
    docs = pd.read_parquet(config.out("corpus", "docs.parquet"),
                           columns=["doc_id", "date", "domain", "in_analysis", "is_press_release", "is_english",
                                    "is_canonical"])
    lab = pd.read_parquet(config.out("industry", "doc_labels_sample.parquet"))
    pred = pd.read_parquet(config.out("industry", "doc_industry_pred.parquet"))
    pairs = pd.read_parquet(config.out("sentiment", "predictions.parquet"),
                            columns=["entity_id", "doc_id", "block_uid", "date", "domain", "is_press_release",
                                     "p_positive", "p_negative", "p_mixed", "pred", "sample_weight"])
    ents = pd.read_parquet(config.out("entities", "entities.parquet"))
    topics = pd.read_csv(config.out("topics", "topic_summary.csv"))
    block_topics = pd.read_parquet(config.out("topics", "block_topics.parquet"), columns=["block_uid", "topic"])
    doc_topics = pd.read_parquet(config.out("topics", "doc_topic_shares.parquet"), columns=["doc_id", "topic"])

    expo = industry_exposure(lab)
    monthly, change = industry_trend(pred, docs)
    direction = direction_by_industry(lab, pairs, pred)
    mech = mechanism_by_industry(lab)
    ranks = entity_rankings(pairs, ents)
    strend = sentiment_trend(pairs)
    factors = factor_table(lab)
    fac_ind = factor_by_industry(lab)
    outcomes = outcome_by_industry(lab)
    topic_q = topic_quarterly_share(doc_topics, docs, topics)
    topic_sent = topic_sentiment(pairs, block_topics, topics)
    ent_cov = entity_coverage(ents)
    top_adopting = [r for r in expo[~expo["supply_side"] & (expo["industry"] != NO_INDUSTRY)]["industry"].head(5)]
    ind_q = industry_quarterly_sentiment(pairs, pred, top_adopting)
    ent_q = quarterly_sentiment(pairs, pairs["entity_id"], TREND_COMPANIES)

    for name, df in [("industry_exposure.csv", expo), ("industry_monthly_share.csv", monthly),
                     ("industry_share_change.csv", change), ("industry_direction.csv", direction),
                     ("industry_mechanisms.csv", mech), ("entity_sentiment.csv", ranks),
                     ("sentiment_monthly.csv", strend), ("adoption_factors.csv", factors),
                     ("adoption_factors_by_industry.csv", fac_ind), ("adoption_outcomes_by_industry.csv", outcomes),
                     ("topic_quarterly_share.csv", topic_q), ("topic_sentiment.csv", topic_sent),
                     ("entity_coverage.csv", ent_cov), ("industry_quarterly_sentiment.csv", ind_q),
                     ("entity_quarterly_sentiment.csv", ent_q)]:
        df.to_csv(tables_path(name), index=False)

    corpus = _load(config.out("corpus", "corpus_stats.json"))
    topics = _load(config.out("topics", "topic_model_stats.json"))
    sent_eval = _load(config.out("sentiment", "test_metrics.json"))
    ind_eval = _load(config.out("industry", "industry_classifier_metrics.json"))
    ent_stats = _load(config.out("entities", "entity_stats.json"))
    mech_all = mech[mech["industry"] == "all_sector_docs"].iloc[0]

    results = {
        "corpus": {k: corpus[k] for k in ["input_docs", "analysis_docs", "analysis_blocks", "non_english_docs",
                                          "non_canonical_docs", "date_min", "date_max", "press_release_share_analysis"]},
        "topics": topics,
        "entities": {k: ent_stats[k] for k in ["final_entities", "final_entities_ge30_docs", "reviewed", "review_dropped"]},
        "sentiment_eval": {k: sent_eval.get(k) for k in ["n_test", "macro_f1", "macro_f1_by_type", "labeler_drift"]},
        "sentiment_pairs_scored": int(len(pairs)),
        "doc_sample_n": int(len(lab)),
        "industry_classifier_f1": {k: v.get("test_f1") for k, v in ind_eval.items()},
        "rq1_top_adopting_industries": expo[~expo["supply_side"] & (expo["industry"] != NO_INDUSTRY)].head(6)
            [["industry", "share_any", "share_any_lo", "share_any_hi"]].to_dict("records"),
        "rq1_supply_side": expo[expo["supply_side"]][["industry", "share_any", "share_any_lo", "share_any_hi"]].to_dict("records"),
        "rq1_no_industry_share": float(expo.loc[expo["industry"] == NO_INDUSTRY, "share_primary"].iloc[0]),
        "rq1_fastest_growing": change.head(5).to_dict("records"),
        "rq2_direction": direction.to_dict("records"),
        "rq2_mechanisms_all_sector_docs": {m: float(mech_all[f"share_{m}"]) for m in MECHANISMS},
        "rq2_top_entities": ranks.head(40)[["name", "type", "n_docs_scored", "sentiment_index", "sentiment_index_lo",
                                            "sentiment_index_hi", "sentiment_index_excl_press"]].to_dict("records"),
        "rq3_factors": factors.to_dict("records"),
        "outcome_shares_all": lab["outcome"].value_counts(normalize=True).to_dict(),
        "topic_sentiment_most_negative": topic_sent.head(5)[["label", "n_docs_scored", "sentiment_index"]].to_dict("records"),
        "topic_sentiment_most_positive": topic_sent.tail(5)[["label", "n_docs_scored", "sentiment_index"]].to_dict("records"),
    }
    write_json(results, config.out("results.json"))
    log.info("wrote %d tables and results.json", 15)


if __name__ == "__main__":
    main()
