"""Static figures for the notebooks and the deck, drawn from the s60 tables."""

from __future__ import annotations

import json

import matplotlib
import matplotlib.dates

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from . import config
from .common import setup_logging
from .taxonomy import FACTORS, INDUSTRIES, MECHANISMS

log = setup_logging("s70_figures")

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_3 = "#8a8983"
GRID = "#e6e5e1"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
MAGENTA = "#e87ba4"
RED = "#e34948"
NEUTRAL = "#d9d8d3"
NEUTRAL_2 = "#efeeea"
CATEGORICAL = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA]
BLUES = LinearSegmentedColormap.from_list("blues", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])

SHORT = {
    "software_it": "Software & IT", "semiconductors_hardware": "Semiconductors & hardware",
    "telecom": "Telecom", "finance_insurance": "Finance & insurance",
    "healthcare_life_sciences": "Healthcare & life sciences", "education": "Education",
    "media_entertainment": "Media & entertainment", "retail_consumer": "Retail & consumer",
    "manufacturing_industrial": "Manufacturing & industrial", "transportation_automotive": "Transport & automotive",
    "energy_utilities": "Energy & utilities", "legal_professional": "Legal & professional services",
    "government_defense": "Government & defense", "agriculture_food": "Agriculture & food",
    "real_estate_construction": "Real estate & construction", "none": "No specific industry",
}
MECH_SHORT = {"automation": "Automation", "augmentation": "Augmentation", "cost_efficiency": "Cost & efficiency",
              "new_offerings": "New offerings", "workflow_redesign": "Workflow redesign"}
FACTOR_SHORT = {"data": "Data", "cost_roi": "Cost / ROI", "regulation_policy": "Regulation & policy",
                "talent_skills": "Talent & skills", "integration": "Integration", "trust_safety": "Trust & safety",
                "compute_infrastructure": "Compute & infrastructure", "partnerships": "Partnerships",
                "leadership_strategy": "Leadership & strategy"}


def style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": ["Inter", "Helvetica Neue", "Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2,
        "ytick.color": INK_2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
        "axes.titlesize": 12, "axes.titleweight": "semibold", "axes.titlecolor": INK,
        "axes.titlelocation": "left", "axes.titlepad": 12, "legend.frameon": False,
        "xtick.major.size": 0, "ytick.major.size": 0, "lines.linewidth": 2, "axes.axisbelow": True,
    })


def fig_path(name: str):
    return config.out("figures", name)


def save(fig, name: str, note: str | None = None, note_y: float = -0.02) -> None:
    if note:
        fig.text(0.01, note_y, note, fontsize=8, color=INK_3, ha="left", va="top")
    fig.savefig(fig_path(name), dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("saved %s", name)


def table(name: str) -> pd.DataFrame:
    return pd.read_csv(config.out("tables", name))


def pct(x) -> str:
    return f"{100 * x:.0f}%" if x >= 0.095 else f"{100 * x:.1f}%"


def spread_labels(ys: list[float], min_gap: float) -> list[float]:
    """Nudge end-of-line label positions apart so they do not overlap."""
    order = np.argsort(ys)
    out = np.array(ys, dtype=float)
    for a, b in zip(order[:-1], order[1:]):
        if out[b] - out[a] < min_gap:
            out[b] = out[a] + min_gap
    return out.tolist()


def dot_ci(ax, y, est, lo, hi, color, size=38):
    ax.hlines(y, lo, hi, color=color, linewidth=2, alpha=0.45, zorder=2)
    ax.scatter(est, y, s=size, color=color, zorder=3, edgecolor=SURFACE, linewidth=1.5)


# ---------------------------------------------------------------------------------------

def corpus_funnel(results: dict) -> None:
    c = results["corpus"]
    stages = [("Raw articles", 199_989), ("AI-relevant after block + sentence filters", c["input_docs"]),
              ("English (language ID)", c["input_docs"] - c["non_english_docs"]),
              ("Analysis set: de-syndicated, non-boilerplate", c["analysis_docs"])]
    fig, ax = plt.subplots(figsize=(8, 2.8))
    y = np.arange(len(stages))[::-1]
    vals = [v for _, v in stages]
    ax.barh(y, vals, color=[NEUTRAL, NEUTRAL, NEUTRAL, BLUE], height=0.62)
    for yi, (lab, v) in zip(y, stages):
        ax.text(v + 2500, yi, f"{v:,}", va="center", color=INK, fontsize=10)
    ax.set_yticks(y, [s for s, _ in stages])
    ax.set_xlim(0, max(vals) * 1.15)
    ax.xaxis.set_visible(False)
    ax.grid(False)
    ax.set_title(f"From {stages[0][1] / 1000:.0f}K scraped articles to {c['analysis_docs'] / 1000:.0f}K distinct AI stories")
    save(fig, "f01_corpus_funnel.png",
         f"Near-duplicate clusters (MinHash, Jaccard ≥ 0.8) collapse {c['non_canonical_docs']:,} syndicated copies.")


def industry_exposure() -> None:
    full = table("industry_exposure.csv")
    supply = full[full["supply_side"]]
    t = full[~full["supply_side"] & (full["industry"] != "none")].sort_values("share_any")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    y = np.arange(len(t))
    for yi, r in zip(y, t.itertuples()):
        dot_ci(ax, yi, r.share_any, r.share_any_lo, r.share_any_hi, BLUE)
        ax.text(r.share_any_hi + 0.004, yi, pct(r.share_any), va="center", fontsize=8.5, color=INK_2)
    ax.set_yticks(y, [SHORT[k] for k in t["industry"]])
    ax.set_xlim(0, t["share_any_hi"].max() * 1.12)
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.02))
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Share of AI news articles that discuss the industry (95% CI)")
    ax.grid(axis="y", visible=False)
    ax.set_title("Which AI-adopting industries the news covers")
    none = full.set_index("industry").loc["none", "share_primary"]
    sup = "; ".join(f"{SHORT[r.industry]} {pct(r.share_any)}" for r in supply.itertuples())
    save(fig, "f02_industry_exposure.png",
         f"LLM-labeled random sample of 15K articles; up to two industries per article. Not shown: AI supply side ({sup}); "
         f"{pct(none)} name no industry.")


def industry_change() -> None:
    t = table("industry_share_change.csv").sort_values("change_pp")
    fig, ax = plt.subplots(figsize=(8, 5.2))
    y = np.arange(len(t))
    for yi, r in zip(y, t.itertuples()):
        c = BLUE if r.change_pp >= 0 else RED
        dot_ci(ax, yi, r.change_pp, r.change_pp_lo, r.change_pp_hi, c)
    ax.axvline(0, color=INK_3, linewidth=0.8)
    ax.set_yticks(y, [SHORT[k] for k in t["industry"]])
    ax.set_xlabel("Change in share of articles, percentage points (95% CI)")
    ax.grid(axis="y", visible=False)
    ax.set_title("Where coverage moved: before ChatGPT vs. the last 12 months")
    save(fig, "f03_industry_change.png", "All 125K analysis articles, industry assigned by a classifier trained on the LLM-labeled sample.")


def industry_trend(results: dict) -> None:
    m = table("industry_monthly_share.csv")
    m["month"] = pd.to_datetime(m["month"])
    top = [r["industry"] for r in results["rq1_top_adopting_industries"][:5]]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ends = []
    for k, c in zip(top, CATEGORICAL):
        s = m.set_index("month")[k].rolling(3, min_periods=2).mean()
        ax.plot(s.index, s.values, color=c, label=SHORT[k])
        ends.append((s.index[-1], s.values[-1], SHORT[k]))
    lo, hi = ax.get_ylim()
    for (x, _, lab), yl in zip(ends, spread_labels([e[1] for e in ends], (hi - lo) * 0.045)):
        ax.text(x + pd.Timedelta(days=12), yl, lab, color=INK_2, fontsize=8.5, va="center")
    ax.axvline(pd.Timestamp("2022-11-30"), color=INK_3, linewidth=0.8)
    ax.text(pd.Timestamp("2022-12-10"), lo + (hi - lo) * 0.02, "ChatGPT launch", fontsize=8, color=INK_3, va="bottom")
    ax.set_ylim(lo, hi)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_ylabel("Share of monthly articles (3-month mean)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=8.5)
    ax.set_title("Coverage of the five most-discussed adopting sectors")
    ax.set_xlim(m["month"].min(), m["month"].max() + pd.Timedelta(days=150))
    save(fig, "f04_industry_trend.png")


def direction_by_industry() -> None:
    t = table("industry_direction.csv").sort_values("net_direction")
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    y = np.arange(len(t))
    left = -t["share_negative"].to_numpy()
    ax.barh(y, -left, left=left, color=RED, height=0.62, label="Negative")
    ax.barh(y, t["share_mixed"], left=0, color=NEUTRAL, height=0.62, label="Mixed")
    ax.barh(y, t["share_neutral"], left=t["share_mixed"], color=NEUTRAL_2, height=0.62, label="Neutral / descriptive")
    ax.barh(y, t["share_positive"], left=t["share_mixed"] + t["share_neutral"], color=BLUE, height=0.62, label="Positive")
    ax.axvline(0, color=INK_3, linewidth=0.8)
    ax.set_yticks(y, [SHORT[k] for k in t["industry"]])
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{abs(v) * 100:.0f}%"))
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=4, fontsize=8.5)
    for yi, r in zip(y, t.itertuples()):
        ax.text(1.01, yi, f"net {r.net_direction:+.2f}", transform=ax.get_yaxis_transform(), va="center",
                fontsize=8.5, color=INK_2)
    ax.set_title("How articles describe AI's effect on each industry")
    save(fig, "f05_direction_by_industry.png", "Negative share to the left of zero; net = positive share − negative share. Industries with ≥150 sampled articles.")


def mechanism_heatmap() -> None:
    t = table("industry_mechanisms.csv")
    t = t[t["industry"] != "all_sector_docs"].copy()
    cols = [f"share_{m}" for m in MECHANISMS]
    t = t.sort_values("n_docs_sample", ascending=False)
    z = t[cols].to_numpy()
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(t) + 0.9))
    im = ax.imshow(z, cmap=BLUES, vmin=0, vmax=max(0.5, float(np.nanmax(z))), aspect="auto")
    ax.set_xticks(np.arange(-0.5, z.shape[1]), minor=True)
    ax.set_yticks(np.arange(-0.5, z.shape[0]), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            ax.text(j, i, f"{100 * z[i, j]:.0f}", ha="center", va="center", fontsize=8.5,
                    color="white" if z[i, j] > 0.33 else INK)
    ax.set_xticks(range(len(cols)), [MECH_SHORT[m] for m in MECHANISMS])
    ax.set_yticks(range(len(t)), [SHORT[k] for k in t["industry"]])
    ax.grid(which="major", visible=False)
    ax.xaxis.tick_top()
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.outline.set_visible(False)
    cb.ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_title("Mechanisms named in each industry's AI coverage (% of articles)", pad=28)
    save(fig, "f06_mechanism_heatmap.png", "Share of an industry's sampled articles that state each mechanism; rows are not normalized, so cells compare across industries.")


def entity_sentiment(kind: str, fname: str, title: str, n: int = 20) -> None:
    t = table("entity_sentiment.csv")
    t = t[(t["type"] == kind) & (t["kind"] == "named")].sort_values("n_docs", ascending=False).head(n)
    t = t.sort_values("sentiment_index")
    fig, ax = plt.subplots(figsize=(8, 0.3 * len(t) + 1.2))
    y = np.arange(len(t))
    for yi, r in zip(y, t.itertuples()):
        dot_ci(ax, yi, r.sentiment_index, r.sentiment_index_lo, r.sentiment_index_hi, BLUE if r.sentiment_index >= 0 else RED)
        ax.text(1.01, yi, f"{int(r.n_docs_scored):,}", transform=ax.get_yaxis_transform(), va="center", fontsize=8, color=INK_3)
    ax.text(1.01, len(t) - 0.2, "articles", transform=ax.get_yaxis_transform(), fontsize=8, color=INK_3)
    ax.axvline(0, color=INK_3, linewidth=0.8)
    ax.set_yticks(y, t["name"])
    ax.set_xlabel("Sentiment index: mean P(positive) − P(negative) per article (95% CI)")
    ax.grid(axis="y", visible=False)
    ax.set_title(title)
    save(fig, fname)


def sentiment_trend() -> None:
    t = table("sentiment_monthly.csv")
    t["month"] = pd.to_datetime(t["month"])
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.fill_between(t["month"], t["sentiment_index_lo"], t["sentiment_index_hi"], color=BLUE, alpha=0.15, linewidth=0)
    ax.plot(t["month"], t["sentiment_index"], color=BLUE, label="All articles")
    ax.plot(t["month"], t["sentiment_index_excl_press"], color=ORANGE, linewidth=1.5, label="Excluding press releases")
    ax.axvline(pd.Timestamp("2022-11-30"), color=INK_3, linewidth=0.8)
    lo, hi = ax.get_ylim()
    ax.text(pd.Timestamp("2022-12-10"), lo + (hi - lo) * 0.02, "ChatGPT launch", fontsize=8, color=INK_3, va="bottom")
    ax.set_ylabel("Sentiment index")
    ax.legend(loc="lower right", fontsize=8.5)
    ax.set_title("Entity-level AI-impact sentiment by month")
    save(fig, "f09_sentiment_trend.png", "Each article weighted once; band is the 95% bootstrap interval.")


def adoption_factors() -> None:
    t = table("adoption_factors.csv")
    t["total"] = t["share_enabler"] + t["share_barrier"]
    t = t.sort_values("total")
    fig, ax = plt.subplots(figsize=(8, 4.2))
    y = np.arange(len(t))
    ax.barh(y, -t["share_barrier"], color=RED, height=0.6, label="Cited as a barrier")
    ax.barh(y, t["share_enabler"], color=BLUE, height=0.6, label="Cited as an enabler")
    ax.axvline(0, color=INK_3, linewidth=0.8)
    ax.set_yticks(y, [FACTOR_SHORT[f] for f in t["factor"]])
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{abs(v) * 100:.0f}%"))
    ax.set_xlabel("Share of sampled articles")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", fontsize=8.5)
    ax.set_title("What articles say helps or hinders AI adoption")
    save(fig, "f10_adoption_factors.png")


def setback_rates() -> None:
    t = table("adoption_factors.csv")
    t = t.dropna(subset=["setback_rate_when_barrier"]).sort_values("setback_rate_when_barrier")
    if t.empty:
        return
    base = float(t["setback_rate_all_adoption"].iloc[0])
    fig, ax = plt.subplots(figsize=(8, 0.4 * len(t) + 1.4))
    y = np.arange(len(t))
    for yi, r in zip(y, t.itertuples()):
        dot_ci(ax, yi, r.setback_rate_when_barrier, r.setback_rate_when_barrier_lo, r.setback_rate_when_barrier_hi, RED)
        ax.text(r.setback_rate_when_barrier_hi + 0.01, yi, f"n = {int(r.n_adoption_docs_citing_barrier)}", va="center",
                fontsize=8, color=INK_3)
    ax.axvline(base, color=INK_3, linewidth=0.8)
    ax.text(base, -0.9, f" all adoption stories: {pct(base)}", fontsize=8, color=INK_3, va="center")
    ax.set_ylim(-1.3, len(t) - 0.5)
    ax.set_yticks(y, [FACTOR_SHORT[f] for f in t["factor"]])
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Setback rate among adoption stories (setback / (deployed + setback))")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, min(1.0, t["setback_rate_when_barrier_hi"].max() + 0.12))
    ax.set_title("Setback rate of adoption stories that cite each barrier")
    save(fig, "f11_setback_rates.png", "Dots: setback / (deployed + setback) among stories citing the factor as a barrier, 95% CI. "
         "Associations in news coverage, not causal effects.")


def topics_top(n: int = 20) -> None:
    t = pd.read_csv(config.out("topics", "topic_summary.csv"))
    t = t[(t["topic"] >= 0) & t["is_coherent"].fillna(True)].sort_values("n_docs", ascending=False).head(n)
    t = t.sort_values("n_docs")
    fig, ax = plt.subplots(figsize=(8, 0.3 * len(t) + 1.2))
    y = np.arange(len(t))
    ax.barh(y, t["n_docs"], color=BLUE, height=0.62)
    for yi, v in zip(y, t["n_docs"]):
        ax.text(v, yi, f" {v:,}", va="center", fontsize=8, color=INK_2)
    ax.set_yticks(y, t["label"])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Articles with at least one block in the topic")
    ax.set_title("Largest topics in the corpus (BERTopic, labeled by LLM)")
    save(fig, "f12_topics_top.png")


def sentiment_confusion() -> None:
    m = json.load(open(config.out("sentiment", "test_metrics.json")))
    if "confusion" not in m:
        return
    z = np.array(m["confusion"]["matrix"], dtype=float)
    zr = z / z.sum(1, keepdims=True)
    labels = ["Positive", "Negative", "Mixed / unclear"]
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    ax.imshow(zr, cmap=BLUES, vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{100 * zr[i, j]:.0f}%\n({int(z[i, j])})", ha="center", va="center", fontsize=8.5,
                    color="white" if zr[i, j] > 0.55 else INK)
    ax.set_xticks(range(3), labels)
    ax.set_yticks(range(3), labels)
    ax.set_xlabel("Classifier")
    ax.set_ylabel("LLM label (held-out test)")
    ax.grid(False)
    ax.set_title(f"Sentiment classifier, macro-F1 {m['macro_f1']:.2f} (n = {m['n_test']:,})")
    save(fig, "f13_sentiment_confusion.png")


def topic_trend_heatmap(n: int = 20) -> None:
    q = table("topic_quarterly_share.csv")
    topics = pd.read_csv(config.out("topics", "topic_summary.csv"))
    keep = topics[(topics["topic"] >= 0) & topics["is_coherent"].fillna(True)].sort_values("n_docs", ascending=False).head(n)
    z = q[q["topic"].isin(keep["topic"])].pivot_table(index="topic", columns="quarter", values="share").fillna(0)
    z = z.loc[keep["topic"]]
    labels = keep.set_index("topic").loc[z.index, "label"]
    fig, ax = plt.subplots(figsize=(10, 0.32 * len(z) + 1.4))
    im = ax.imshow(z.to_numpy(), cmap=BLUES, vmin=0, aspect="auto")
    ax.set_xticks(np.arange(-0.5, z.shape[1]), minor=True)
    ax.set_yticks(np.arange(-0.5, z.shape[0]), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=1.5)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="minor", length=0)
    ax.set_yticks(range(len(z)), labels)
    cols = list(z.columns)
    ax.set_xticks(range(0, len(cols), 2), [cols[i] for i in range(0, len(cols), 2)], rotation=0, fontsize=8.5)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.outline.set_visible(False)
    cb.ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_title("Share of each quarter's articles that touch the topic (20 largest topics)")
    save(fig, "f14_topic_trend.png", "An article counts toward a topic if any of its paragraphs is assigned to it; last partial quarter excluded.")


def topic_sentiment(n_each: int = 10) -> None:
    t = table("topic_sentiment.csv")
    t = t[t["is_coherent"].fillna(True)].sort_values("sentiment_index")
    t = pd.concat([t.head(n_each), t.tail(n_each)]).drop_duplicates("topic")
    fig, ax = plt.subplots(figsize=(8, 0.3 * len(t) + 1.2))
    y = np.arange(len(t))
    for yi, r in zip(y, t.itertuples()):
        dot_ci(ax, yi, r.sentiment_index, r.sentiment_index_lo, r.sentiment_index_hi, BLUE if r.sentiment_index >= 0 else RED)
        ax.text(1.01, yi, f"{int(r.n_docs_scored):,}", transform=ax.get_yaxis_transform(), va="center", fontsize=8, color=INK_3)
    ax.text(1.01, len(t) - 0.2, "articles", transform=ax.get_yaxis_transform(), fontsize=8, color=INK_3)
    ax.axhline(n_each - 0.5, color=GRID, linewidth=1)
    ax.axvline(0, color=INK_3, linewidth=0.8)
    ax.set_yticks(y, t["label"])
    ax.set_xlabel("Entity sentiment index within the topic (95% CI)")
    ax.grid(axis="y", visible=False)
    ax.set_title(f"Topics with the most negative and most positive entity sentiment")
    save(fig, "f15_topic_sentiment.png", "Mean over articles of P(positive) − P(negative) for entity mentions in the topic's paragraphs; topics with ≥200 scored articles.")


def entity_coverage() -> None:
    t = table("entity_coverage.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    for ax, kind, title in zip(axes, ["company", "technology"], ["Companies", "Products and technologies"]):
        sub = t[t["type"] == kind].sort_values("n_docs").tail(15)
        y = np.arange(len(sub))
        ax.barh(y, sub["n_docs"], color=BLUE, height=0.62)
        for yi, v in zip(y, sub["n_docs"]):
            ax.text(v, yi, f" {v:,}", va="center", fontsize=8, color=INK_2)
        ax.set_yticks(y, sub["name"])
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, sub["n_docs"].max() * 1.2)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:.0f}K"))
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Articles mentioning the entity")
    fig.suptitle("Most-covered named entities after canonicalization", x=0.02, ha="left", fontsize=12, fontweight="semibold")
    fig.tight_layout()
    save(fig, "f16_entity_coverage.png", "GLiNER mentions (score ≥ 0.50) merged by canonical key and alias table, reviewed by the LLM.")


def _quarter_lines(t: pd.DataFrame, names: dict, title: str, fname: str, note: str, start: str | None = None) -> None:
    t = t.copy()
    t["q"] = pd.PeriodIndex(t["quarter"], freq="Q").to_timestamp()
    if start:
        t = t[t["q"] >= pd.Timestamp(start)]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ends = []
    for (g, sub), c in zip(t.groupby("group", sort=False), CATEGORICAL + ["#4a3aa7", "#008300"]):
        sub = sub.sort_values("q")
        ax.plot(sub["q"], sub["sentiment_index"], color=c, label=names.get(g, g), marker="o", markersize=3.5)
        ends.append((sub["q"].iloc[-1], sub["sentiment_index"].iloc[-1], names.get(g, g)))
    lo, hi = ax.get_ylim()
    for (x, _, lab), yl in zip(ends, spread_labels([e[1] for e in ends], (hi - lo) * 0.05)):
        ax.text(x + pd.Timedelta(days=20), yl, lab, color=INK_2, fontsize=8.5, va="center")
    ax.axvline(pd.Timestamp("2022-11-30"), color=INK_3, linewidth=0.8)
    ax.text(pd.Timestamp("2022-12-10"), lo + (hi - lo) * 0.02, "ChatGPT launch", fontsize=8, color=INK_3, va="bottom")
    ax.set_ylim(lo, hi)
    ax.set_xlim(t["q"].min() - pd.Timedelta(days=20), t["q"].max() + pd.Timedelta(days=240))
    ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator(bymonth=[1, 7]))
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y-%m"))
    ax.set_ylabel("Sentiment index (quarterly)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=8.5)
    ax.set_title(title)
    save(fig, fname, note, note_y=-0.1)


def sentiment_trend_by_group() -> None:
    ind = table("industry_quarterly_sentiment.csv")
    order = ind.groupby("group")["n_docs"].sum().sort_values(ascending=False).index
    ind["group"] = pd.Categorical(ind["group"], order)
    _quarter_lines(ind.sort_values("group"), SHORT, "Entity sentiment by quarter in the five most-covered adopting sectors",
                   "f17_sentiment_trend_industries.png", "Quarters with at least 30 scored articles; industry from the embedding classifier.")
    ent = table("entity_quarterly_sentiment.csv")
    names = dict(zip(table("entity_sentiment.csv")["entity_id"], table("entity_sentiment.csv")["name"]))
    _quarter_lines(ent, names, "Entity sentiment by quarter for the most-covered AI companies",
                   "f18_sentiment_trend_companies.png", "Up to 3,000 random contexts per company; quarters with at least 30 scored articles, from 2022 Q4.",
                   start="2022-10-01")


def main() -> None:
    style()
    results = json.load(open(config.out("results.json")))
    corpus_funnel(results)
    industry_exposure()
    industry_change()
    industry_trend(results)
    direction_by_industry()
    mechanism_heatmap()
    entity_sentiment("company", "f07_entity_sentiment_companies.png", "AI-impact sentiment toward the 20 most-covered companies")
    entity_sentiment("technology", "f08_entity_sentiment_technologies.png", "AI-impact sentiment toward the 20 most-covered products and platforms")
    sentiment_trend()
    adoption_factors()
    setback_rates()
    topics_top()
    sentiment_confusion()
    topic_trend_heatmap()
    topic_sentiment()
    entity_coverage()
    sentiment_trend_by_group()


if __name__ == "__main__":
    main()
