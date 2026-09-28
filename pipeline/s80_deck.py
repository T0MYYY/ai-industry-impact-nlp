"""Results deck. Every number on a slide is read from results.json or the s60 tables."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .deck_layout import Deck
from .s70_figures import FACTOR_SHORT, MECH_SHORT, SHORT


# Short sector names for slide titles and callouts.
NAME = {
    "software_it": "software", "semiconductors_hardware": "chips and hardware", "telecom": "telecom",
    "finance_insurance": "finance", "healthcare_life_sciences": "healthcare", "education": "education",
    "media_entertainment": "media", "retail_consumer": "retail", "manufacturing_industrial": "manufacturing",
    "transportation_automotive": "transport", "energy_utilities": "energy", "legal_professional": "legal services",
    "government_defense": "government", "agriculture_food": "agriculture", "real_estate_construction": "real estate",
}


# Figures reported by the upstream notebooks (01, 02B, 02D, 06A, 06B).
UPSTREAM = {
    "raw_docs": 199_989, "block_labels": 12_000, "block_ai_share": 0.352, "block_f1": 0.862, "block_threshold": 0.58,
    "ai_blocks": 1_184_289, "sentence_labels": 7_302, "sentence_f1": 0.960, "clean_blocks": 731_989,
    "sent_labels": 8_999, "sent_pos": 5174 / 8999, "sent_mixed": 2661 / 8999, "sent_neg": 1164 / 8999, "sent_val_f1": 0.824,
}


def pct(x: float, nd: int = 0) -> str:
    return f"{100 * x:.{nd}f}%"


def load():
    t = lambda n: pd.read_csv(config.out("tables", n))  # noqa: E731
    return {
        "res": json.load(open(config.out("results.json"))),
        "expo": t("industry_exposure.csv"), "change": t("industry_share_change.csv"),
        "dir": t("industry_direction.csv"), "mech": t("industry_mechanisms.csv"),
        "ents": t("entity_sentiment.csv"), "trend": t("sentiment_monthly.csv"),
        "fac": t("adoption_factors.csv"), "topics": pd.read_csv(config.out("topics", "topic_summary.csv")),
        "topic_q": t("topic_quarterly_share.csv"), "topic_sent": t("topic_sentiment.csv"), "ent_cov": t("entity_coverage.csv"),
        "ind_q": t("industry_quarterly_sentiment.csv"), "ent_q": t("entity_quarterly_sentiment.csv"),
    }


def build(out_path: Path) -> None:
    d = load()
    r, expo, change, direc, mech, ents, trend, fac, topics = (
        d["res"], d["expo"], d["change"], d["dir"], d["mech"], d["ents"], d["trend"], d["fac"], d["topics"])
    F = lambda n: config.out("figures", n)  # noqa: E731
    c = r["corpus"]
    se = r["sentiment_eval"]

    adopt = expo[~expo["supply_side"] & (expo["industry"] != "none")].sort_values("share_any", ascending=False)
    supply = expo[expo["supply_side"]]
    supply_share = float(expo.loc[expo["industry"] == "software_it", "share_any"].iloc[0])
    top3 = adopt.head(3)
    ch = change.set_index("industry")
    dsort = direc.sort_values("net_direction", ascending=False)
    rho = float(direc[["net_direction", "entity_sentiment_index"]].corr("spearman").iloc[0, 1])
    m_ind = mech[mech["industry"] != "all_sector_docs"].set_index("industry")
    m_all = mech[mech["industry"] == "all_sector_docs"].iloc[0]
    lead_mech = {k: max(MECH_SHORT, key=lambda m: m_ind.loc[k, f"share_{m}"]) for k in m_ind.index}
    fac_i = fac.set_index("factor")
    base_setback = float(fac["setback_rate_all_adoption"].iloc[0])
    f1s = [v for v in r["industry_classifier_f1"].values() if v is not None]
    named = ents[ents["kind"] == "named"]
    se = dict(se, report=json.load(open(config.out("sentiment", "test_metrics.json")))["report"])
    comp = named[named["type"] == "company"].sort_values("n_docs", ascending=False).head(20)
    tr = trend.assign(year=trend["month"].str[:4])
    yearly = tr.groupby("year").apply(lambda g: np.average(g["sentiment_index"], weights=g["n_docs"]))
    low_month = trend.loc[trend["sentiment_index"].idxmin()]

    deck = Deck()

    # 1 -------------------------------------------------------------------------------
    deck.title_slide(
        "Where AI news says AI is changing industries",
        f"Topic, entity and sentiment evidence from {c['analysis_docs'] / 1000:.0f}K de-duplicated AI news articles, "
        f"{c['date_min'][:4]}–{c['date_max'][:4]}",
        "Chiyang (Tom) Chen  ·  ADSP 32018 Next-Gen NLP  ·  University of Chicago")

    # 2 -------------------------------------------------------------------------------
    reg = fac_i.loc["regulation_policy"]
    deck.stats_slide("Executive summary", [
        (pct(supply_share), "of AI news discusses software & IT",
         f"Among sectors adopting AI, {NAME[top3.iloc[0].industry]} ({pct(top3.iloc[0].share_any)}), "
         f"{NAME[top3.iloc[1].industry]} ({pct(top3.iloc[1].share_any)}) and "
         f"{NAME[top3.iloc[2].industry]} ({pct(top3.iloc[2].share_any)}) get the most coverage."),
        (f"{dsort.iloc[0].net_direction:+.2f}", f"net framing for {NAME[dsort.iloc[0].industry]}",
         f"vs. {dsort.iloc[-1].net_direction:+.2f} for {NAME[dsort.iloc[-1].industry]}; a separately trained entity "
         f"sentiment model ranks industries alike (ρ = {rho:.2f})."),
        (pct(reg.setback_rate_when_barrier), "setback rate when regulation is the barrier",
         f"vs. {pct(base_setback)} for all adoption stories; talent and integration barriers sit below it."),
    ], subtitle="Three research questions, one number each")

    # 3 -------------------------------------------------------------------------------
    deck.columns_slide("Why industry impact is uneven", [
        ("Exposure estimates", [
            "Goldman Sachs (2023): about a quarter of work tasks in the US and Europe could be automated by AI.",
            "Exposure is uneven: office and administrative work, legal, architecture and social sciences above 30%; "
            "construction, installation and building maintenance largely unaffected."]),
        ("Moravec's paradox", [
            "Abstract reasoning is easier to automate than sensorimotor skill.",
            "Implies cognitive, text-heavy sectors are more exposed than physical ones."]),
        ("What this project adds", [
            f"Evidence from {c['analysis_docs'] / 1000:.0f}K AI news articles: which sectors coverage links to AI, "
            "in which direction, through which mechanism, and under what conditions adoption stalls."]),
    ], subtitle="The brief: identify impacted industries and companies, how they are impacted, and what makes adoption succeed")
    # 4 -------------------------------------------------------------------------------
    deck.columns_slide("Three questions about AI's industry impact", [
        ("1  Which industries?", [
            "Share of articles that discuss each industry, from an LLM-labeled random sample",
            "Change in share before ChatGPT vs. the last 12 months, on the full corpus",
            "Supply side (software, chips) separated from adopting sectors"]),
        ("2  Which direction, and how?", [
            "Article-level framing of AI's effect on the industry",
            "Entity-level sentiment from a fine-tuned roberta-large classifier",
            "Mechanisms: automation, augmentation, cost, new offerings, workflow redesign"]),
        ("3  What makes adoption work?", [
            "Adoption outcome per article: deployed, pilot, setback, product launch",
            "Enablers and barriers named in the article",
            "Setback rate among adoption stories that cite each barrier"]),
    ], subtitle="Measured on news coverage, which reflects what is reported, not realized economic effects")

    # 5 -------------------------------------------------------------------------------
    deck.flow_slide("From 200K scraped pages to a de-duplicated analysis corpus", [
        ("AI filter", "Block and sentence classifiers (DistilRoBERTa, DeepSeek labels) keep AI content"),
        ("De-duplicate", f"Language ID and MinHash LSH remove {c['non_canonical_docs']:,} syndicated copies"),
        ("Topics", f"BERTopic on {c['analysis_blocks'] / 1000:.0f}K blocks, GPU UMAP + HDBSCAN"),
        ("Entities", f"GLiNER mentions, alias merge and LLM review of the top 15K entities"),
        ("Sentiment", f"roberta-large scores {r['sentiment_pairs_scored'] / 1000:.0f}K entity–context pairs"),
        ("Industry", f"LLM labels on {r['doc_sample_n'] / 1000:.0f}K sampled articles, classifier for the rest"),
    ], fig=F("f01_corpus_funnel.png"), source="LLM labeling: deepseek-flash, temperature 0, JSON output.")

    # 6 -------------------------------------------------------------------------------
    deck.stats_slide("Cleaning and filtering: from web crawl to AI content", [
        (f"{UPSTREAM['block_f1']:.2f}", "block classifier F1",
         f"DistilRoBERTa on {UPSTREAM['block_labels']:,} DeepSeek-labeled paragraphs ({pct(UPSTREAM['block_ai_share'])} AI); "
         f"threshold {UPSTREAM['block_threshold']}; keeps {UPSTREAM['ai_blocks'] / 1e6:.2f}M AI paragraphs."),
        (f"{UPSTREAM['sentence_f1']:.2f}", "sentence classifier F1",
         f"separates article content from boilerplate on {UPSTREAM['sentence_labels']:,} labeled sentences; "
         f"rebuilds {UPSTREAM['clean_blocks'] / 1000:.0f}K clean paragraphs."),
        (f"{c['non_canonical_docs'] / 1000:.1f}K", "syndicated copies removed",
         "MinHash LSH on word 5-shingles (Jaccard ≥ 0.8); the earliest copy of each story is kept."),
        (f"{c['non_english_docs']}", "non-English pages removed",
         "fastText language ID, although the source field labels every page English."),
    ], subtitle=f"{UPSTREAM['raw_docs']:,} crawled articles → {c['input_docs']:,} with AI content → {c['analysis_docs']:,} distinct stories",
       source="F1 on the validation split of the LLM labels, where the threshold was also chosen (02B, 02D).")
    # 7 -------------------------------------------------------------------------------
    tt = topics[(topics["topic"] >= 0) & topics["is_coherent"].fillna(True)].sort_values("n_docs", ascending=False)
    deck.figure_slide(
        "The largest single theme is AI policy; sector themes are smaller but distinct",
        F("f12_topics_top.png"),
        notes=[f"{r['topics']['n_topics']} topics over {c['analysis_blocks']:,} paragraphs; labels written by the LLM "
               "from keywords and sample passages.",
               f"Largest: {tt.iloc[0].label} ({tt.iloc[0].n_docs:,} articles), {tt.iloc[1].label} "
               f"({tt.iloc[1].n_docs:,}), {tt.iloc[2].label} ({tt.iloc[2].n_docs:,}).",
               "Sector topics such as healthcare, lending, education and entertainment each hold 3–9K articles."],
        source="BERTopic with all-mpnet-base-v2 embeddings; an article counts toward every topic it has a paragraph in.")

    # 8 -------------------------------------------------------------------------------
    tq = d["topic_q"]
    piv = tq.pivot_table(index="label", columns="quarter", values="share")
    delta = (piv.iloc[:, -4:].mean(axis=1) - piv.iloc[:, :3].mean(axis=1)).sort_values()
    deck.figure_slide(
        "Healthcare led early AI coverage; chatbot and AI-investment topics grew after ChatGPT",
        F("f14_topic_trend.png"),
        notes=[f"{delta.index[0]}: {100 * delta.iloc[0]:+.0f} pp of articles, 2022 Q1–Q3 vs. the last four full quarters.",
               f"Largest gains: {delta.index[-1]} ({100 * delta.iloc[-1]:+.0f} pp), {delta.index[-2]} ({100 * delta.iloc[-2]:+.0f} pp).",
               "The chatbot topic peaks in 2023 Q1, the quarter after ChatGPT's release."],
        source="Share of each quarter's articles with at least one paragraph in the topic.")
    # 9 -------------------------------------------------------------------------------
    ec = d["ent_cov"].sort_values("n_docs", ascending=False)
    st = r["entities"]
    deck.figure_slide(
        f"{ec.iloc[0]['name']}, {ec[ec.type == 'technology'].iloc[0]['name']} and Google lead entity coverage by a wide margin",
        F("f16_entity_coverage.png"),
        notes=[f"GLiNER (large-v2.1) finds 5.1M mentions; {st['final_entities_ge30_docs']:,} canonical entities appear in 30+ articles.",
               f"The LLM reviewed the 15K most-covered candidates and dropped {st['review_dropped']:,} vague or broken ones.",
               f"{ec[ec.type == 'technology'].iloc[0]['name']} alone merges {int(ec[ec.type == 'technology'].iloc[0].n_keys_merged)} spelling variants."],
        source="Article counts after canonicalization (suffix stripping, alias table, LLM review).")
    # 10 ------------------------------------------------------------------------------
    deck.figure_slide(
        "A fine-tuned roberta-large scores AI impact toward each entity in context",
        F("f13_sentiment_confusion.png"),
        notes=[f"Training data: {UPSTREAM['sent_labels']:,} entity–context pairs labeled by DeepSeek "
               f"({pct(UPSTREAM['sent_pos'])} positive, {pct(UPSTREAM['sent_mixed'])} mixed, {pct(UPSTREAM['sent_neg'])} negative).",
               "Input: target entity, type, title, source, instruction and a 1,400-character window; split by entity, so no entity is in both train and validation.",
               f"Validation macro-F1 {UPSTREAM['sent_val_f1']:.2f}; fresh test set {se['macro_f1']:.2f}. The model over-predicts negative "
               f"(precision {se['report']['negative']['precision']:.2f})."],
        source="Classifier trained in 06B; test labels from deepseek-flash on pairs never used in training (06C).")
    # 11 ------------------------------------------------------------------------------
    drift = se["labeler_drift"]
    deck.stats_slide("How much to trust the labels", [
        (f"{se['macro_f1']:.2f}", "sentiment macro-F1",
         f"on {se['n_test']:,} fresh entity–context pairs labeled by the LLM, never seen in training."),
        (pct(drift["agreement"]), "labeler agreement",
         f"between deepseek-chat (06A labels) and deepseek-flash on {drift['n']} validation items (κ = {drift['cohen_kappa']:.2f})."),
        (f"{np.median(f1s):.2f}", "median industry F1",
         f"of the embedding classifier vs. held-out LLM labels (range {min(f1s):.2f}–{max(f1s):.2f})."),
        (pct(r["topics"]["outlier_share_after_reduction"], 1), "topic outliers",
         f"after embedding-based outlier reassignment ({pct(r['topics']['outlier_share_before_reduction'])} before)."),
    ], subtitle="All model checks are against held-out LLM labels; there is no human-annotated gold set")

    # 12 ------------------------------------------------------------------------------
    deck.figure_slide(
        f"Among adopting sectors, {NAME[top3.iloc[0].industry]}, {NAME[top3.iloc[1].industry]} "
        f"and {NAME[top3.iloc[2].industry]} get the most AI coverage",
        F("f02_industry_exposure.png"),
        notes=[f"Half of all AI articles ({pct(supply_share)}) are about software and IT; "
               f"semiconductors add {pct(float(supply[supply.industry == 'semiconductors_hardware'].share_any.iloc[0]))}.",
               f"{pct(r['rq1_no_industry_share'])} of articles discuss AI without naming any sector.",
               f"Excluding press releases moves no sector's share by more than "
               f"{100 * (adopt['share_any'] - adopt['share_any_excl_press']).abs().max():.1f} points."],
        source=f"LLM labels on a random sample of {r['doc_sample_n']:,} articles; 95% bootstrap intervals.")

    # 13 ------------------------------------------------------------------------------
    deck.figure_slide(
        "After ChatGPT, coverage moved to the AI supply side and away from healthcare",
        F("f03_industry_change.png"),
        notes=[f"Software & IT {ch.loc['software_it', 'change_pp']:+.0f} pp and semiconductors "
               f"{ch.loc['semiconductors_hardware', 'change_pp']:+.1f} pp of all articles.",
               f"Healthcare fell from {pct(ch.loc['healthcare_life_sciences', 'share_pre_chatgpt'])} to "
               f"{pct(ch.loc['healthcare_life_sciences', 'share_last_12m'])}; transport and manufacturing also shrank.",
               "Shares, not volumes: the corpus grew several-fold after late 2022."],
        source="All analysis articles; industry from the embedding classifier. Pre-ChatGPT = before 2022-11-30.")

    # 14 ------------------------------------------------------------------------------
    deck.figure_slide(
        f"AI is framed most positively for {NAME[dsort.iloc[0].industry]} and "
        f"{NAME[dsort.iloc[1].industry]}, least for {NAME[dsort.iloc[-2].industry]} and "
        f"{NAME[dsort.iloc[-1].industry]}",
        F("f05_direction_by_industry.png"),
        notes=[f"{SHORT[dsort.iloc[-1].industry]}: {pct(dsort.iloc[-1].share_negative)} of articles negative, "
               f"the highest of any sector; {NAME[dsort.iloc[0].industry]}: {pct(dsort.iloc[0].share_negative)}.",
               f"The entity-sentiment classifier (06C), a different model and unit, ranks industries the same way (Spearman ρ = {rho:.2f}).",
               "Neutral means the article reports facts without a stated effect."],
        source="Article framing from LLM labels; industries with at least 150 sampled articles.")

    # 15 ------------------------------------------------------------------------------
    adopting = [k for k in m_ind.index if k not in ("software_it", "semiconductors_hardware")]
    gap = {k: m_ind.loc[k, "share_automation"] - m_ind.loc[k, "share_augmentation"] for k in adopting}
    auto_lead = [k for k, v in sorted(gap.items(), key=lambda x: -x[1]) if v > 0.05]
    aug_lead = [k for k, v in sorted(gap.items(), key=lambda x: x[1]) if v < -0.05]
    names = lambda ks: ", ".join(NAME[k] for k in ks[:-1]) + (" and " if len(ks) > 1 else "") + NAME[ks[-1]]  # noqa: E731
    deck.figure_slide(
        f"Automation outweighs augmentation in {names(auto_lead[:3])}; augmentation leads in {names(aug_lead[:3])}",
        F("f06_mechanism_heatmap.png"),
        notes=[f"Across sector articles, new offerings are named most ({pct(m_all.share_new_offerings)}), then "
               f"augmentation ({pct(m_all.share_augmentation)}) and automation ({pct(m_all.share_automation)}).",
               f"Automation exceeds augmentation by more than 5 points in {len(auto_lead)} adopting sectors, "
               f"the reverse holds in {len(aug_lead)}.",
               f"Energy is the cost story: {pct(m_ind.loc['energy_utilities', 'share_cost_efficiency'])} cite cost or efficiency."],
        source="Share of each industry's sampled articles that state the mechanism; articles can name several.")

    # 16 ------------------------------------------------------------------------------
    cs = comp.sort_values("sentiment_index", ascending=False)
    deck.figure_slide(
        "Infrastructure vendors get the warmest coverage; frontier labs and controversy-linked firms the coolest",
        F("f07_entity_sentiment_companies.png"),
        notes=[f"Highest: {', '.join(cs.head(3)['name'])} (index {cs.iloc[0].sentiment_index:.2f} to {cs.iloc[2].sentiment_index:.2f}).",
               f"OpenAI {named.set_index('name').loc['OpenAI', 'sentiment_index']:.2f}, "
               f"Google {named.set_index('name').loc['Google', 'sentiment_index']:.2f}: the most-covered firms sit near the bottom.",
               f"Below zero: {', '.join(cs[cs.sentiment_index < 0]['name'])}."],
        source="Sentiment index = mean P(positive) − P(negative) per article; up to 3,000 random contexts per entity.")

    # 17 ------------------------------------------------------------------------------
    ts = d["topic_sent"]
    ts = ts[ts["is_coherent"].fillna(True)].sort_values("sentiment_index")
    deck.figure_slide(
        "Security, misinformation and legal topics carry the most negative sentiment",
        F("f15_topic_sentiment.png"),
        notes=[f"Most negative: {ts.iloc[0].label} ({ts.iloc[0].sentiment_index:.2f}), {ts.iloc[1].label} "
               f"({ts.iloc[1].sentiment_index:.2f}), {ts.iloc[2].label} ({ts.iloc[2].sentiment_index:.2f}).",
               f"Most positive: {ts.iloc[-1].label} ({ts.iloc[-1].sentiment_index:.2f}), {ts.iloc[-2].label}, {ts.iloc[-3].label}.",
               f"{len(ts)} coherent topics with at least 200 scored articles."],
        source="Entity-level sentiment of mentions in each topic's paragraphs, averaged per article; 95% bootstrap intervals.")
    # 18 ------------------------------------------------------------------------------
    deck.figure_slide(
        "Entity sentiment dropped after ChatGPT's launch and has stayed below its 2022 level",
        F("f09_sentiment_trend.png"),
        notes=[f"Article-weighted index: {yearly.get('2022', np.nan):.2f} in 2022, {yearly.get('2023', np.nan):.2f} in 2023, "
               f"{yearly.get('2025', np.nan):.2f} in 2025.",
               "Press releases run slightly warmer, but excluding them does not change the pattern.",
               f"Lowest month: {low_month.month[:7]} ({low_month.sentiment_index:.2f})."],
        source="Monthly mean over articles of entity-level sentiment; 95% bootstrap band.")

    # 19 ------------------------------------------------------------------------------
    iq = d["ind_q"]
    edu = iq[iq.group == "education"].set_index("quarter")["sentiment_index"]
    med = iq[iq.group == "media_entertainment"].set_index("quarter")["sentiment_index"]
    deck.figure_slide(
        "Education sentiment collapsed after ChatGPT and recovered; media sentiment slid until 2024",
        F("f17_sentiment_trend_industries.png"),
        notes=[f"Education fell from {edu.get('2022Q4', np.nan):.2f} to {edu.get('2023Q1', np.nan):.2f} in 2023 Q1, then recovered to about {edu.iloc[-4:].mean():.2f}.",
               f"Media slid from {med.iloc[0]:.2f} in 2022 Q1 to {med.min():.2f} at its low and ends at {med.iloc[-1]:.2f}.",
               "Healthcare stays the most positive sector through most of the period."],
        source="Quarterly article-level index; industry from the embedding classifier; quarters with ≥ 30 scored articles.")
    # 20 ------------------------------------------------------------------------------
    eq = d["ent_q"]
    avg = eq[eq["quarter"] >= "2023Q1"].groupby("group")["sentiment_index"].mean().sort_values()
    nm = dict(zip(ents["entity_id"], ents["name"]))
    deck.figure_slide(
        f"Among AI companies, {nm.get(avg.index[-1])} stays the most positive; {nm.get(avg.index[0])} and "
        f"{nm.get(avg.index[1])} the least",
        F("f18_sentiment_trend_companies.png"),
        notes=[f"Average index since 2023: {', '.join(f'{nm.get(g, g)} {v:.2f}' for g, v in avg.sort_values(ascending=False).items())}.",
               "Single quarters move by ±0.1 or more; compare averages rather than individual points."],
        source="Up to 3,000 random contexts per company; quarters with ≥ 30 scored articles.")
    # 21 ------------------------------------------------------------------------------
    fb = fac.sort_values("share_barrier", ascending=False)
    fe = fac.sort_values("share_enabler", ascending=False)
    deck.figure_slide(
        f"{FACTOR_SHORT[fb.iloc[0].factor]} and {FACTOR_SHORT[fb.iloc[1].factor].lower()} are the most-cited barriers; "
        f"{FACTOR_SHORT[fe.iloc[0].factor].lower()} the most-cited enabler",
        F("f10_adoption_factors.png"),
        notes=[f"{FACTOR_SHORT[fb.iloc[0].factor]} is named as a barrier in {pct(fb.iloc[0].share_barrier)} of articles, "
               f"{FACTOR_SHORT[fb.iloc[1].factor].lower()} in {pct(fb.iloc[1].share_barrier)}.",
               f"{FACTOR_SHORT[fe.iloc[0].factor]} appear almost only as enablers ({pct(fe.iloc[0].share_enabler)}).",
               f"Shares are of all {r['doc_sample_n']:,} sampled articles; up to four factors per article."],
        source="LLM labels on the article sample; up to four factors per article.")

    # 22 ------------------------------------------------------------------------------
    sb = fac.dropna(subset=["setback_rate_when_barrier"]).sort_values("setback_rate_when_barrier", ascending=False)
    low = sb[sb["setback_rate_when_barrier_hi"] < base_setback]
    high = sb[sb["setback_rate_when_barrier_lo"] > base_setback]
    fnames = lambda fs: ", ".join(FACTOR_SHORT[f].lower() for f in fs[:-1]) + (" and " if len(fs) > 1 else "") + FACTOR_SHORT[fs[-1]].lower()  # noqa: E731
    deck.figure_slide(
        f"Barriers of {fnames(list(high['factor']))} coincide with setbacks; {fnames(list(low['factor']))} barriers do not",
        F("f11_setback_rates.png"),
        notes=[f"Among adoption stories, {pct(base_setback)} report a setback overall.",
               f"When the barrier is {FACTOR_SHORT[sb.iloc[0].factor].lower()}: {pct(sb.iloc[0].setback_rate_when_barrier)}; "
               f"{FACTOR_SHORT[sb.iloc[1].factor].lower()}: {pct(sb.iloc[1].setback_rate_when_barrier)}.",
               f"{fnames(list(low['factor'])).capitalize()} barriers sit below the baseline: they are reported alongside "
               "deployments that go ahead."],
        source="Setback rate = setback / (deployed + setback). Associations in coverage, not causal effects.")

    # 23 ------------------------------------------------------------------------------
    deck.columns_slide("Takeaways and limits", [
        ("What the coverage shows", [
            f"AI news is mostly about the AI industry itself; among adopters, {NAME[top3.iloc[0].industry]}, "
            f"{NAME[top3.iloc[1].industry]} and {NAME[top3.iloc[2].industry]} lead.",
            f"Framing is positive almost everywhere; {NAME[dsort.iloc[-1].industry]} and "
            f"{NAME[dsort.iloc[-2].industry]} are the most contested sectors, followed by {NAME[dsort.iloc[-3].industry]}.",
            "Setbacks coincide with regulation, leadership and trust barriers far more often than with "
            "skills or integration barriers.",
            f"In line with Goldman Sachs and Moravec: legal coverage is automation-heavy "
            f"({pct(m_ind.loc['legal_professional', 'share_automation'])} cite automation), while construction and "
            f"agriculture are the least-covered sectors (under 1% of articles each)."]),
        ("What it cannot show", [
            "News reflects what is reported and promoted, not realized productivity or job effects.",
            "Labels come from one LLM family; model scores are agreement with it, not human judgment.",
            "Industry shares outside the labeled sample depend on a classifier (F1 0.49–0.84).",
            "The upstream AI filter keeps 91% of documents, so some off-topic text remains."]),
    ], dark=True)

    deck.save(out_path)


def main() -> None:
    out = config.out("deck", "AI_Impact_on_Industries.pptx")
    build(out)
    print(out)


if __name__ == "__main__":
    main()
