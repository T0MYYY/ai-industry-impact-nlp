"""Entity table from the GLiNER mentions of 05A.

GLiNER is not re-run; its 5.1M raw mentions are filtered to the s10 analysis blocks at
one recorded score threshold. Canonicalization keeps the most frequent surface form as
the display name (no title-casing), strips legal suffixes, applies a small alias table,
and resolves type by score-weighted vote across all mentions of the key. The most
frequent entities are then reviewed by the LLM, which can drop, retype or rename them;
renames that collide with another entity merge the two.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, llm
from .common import normalize_spaces, setup_logging, timed, truncate, write_json

log = setup_logging("s30_entities")

MIN_SCORE = 0.50
REVIEW_TOP_N = 15000
MODEL_TYPES = ["company", "technology", "government_institution", "person"]
LABEL_TO_TYPE = {
    "company": "company",
    "organization": "organization",
    "technology": "technology",
    "product": "technology",
    "government institution": "government_institution",
    "government_institution": "government_institution",
    "person": "person",
}
LEGAL_SUFFIX_RE = re.compile(r"\s+(inc|corp|corporation|co|ltd|limited|llc|plc|gmbh|ag|sa|holdings)$")
ALIASES = pd.read_csv(Path(__file__).parent / "data" / "entity_aliases.csv")


def entity_key(text: str) -> str:
    x = normalize_spaces(text).lower()
    x = re.sub(r"['’]s$", "", x)
    x = re.sub(r"['’]", " ", x)
    x = re.sub(r"[\(\)\[\]\{\}\.,;:!?\"“”/\\|®™]+", " ", x)
    x = re.sub(r"[-_]+", " ", x)
    x = re.sub(r"^the\s+", "", re.sub(r"\s+", " ", x).strip())
    if " " in x:
        x = LEGAL_SUFFIX_RE.sub("", x)
    return x.strip()


ALIAS_MAP = {entity_key(a): n for a, n in zip(ALIASES["alias_key"], ALIASES["canonical_name"])}
NUMERIC_JUNK_RE = re.compile(r"^[\d\s$€£%.,:+\-–—/#]+$|^\$|^#\d+|.*\d{1,3}(?:,\d{3})+")


REVIEW_SYSTEM = """You review named-entity candidates extracted from AI news articles.
Return one JSON object:
{"keep": true|false, "type": "company|technology|government_institution|person|other_organization",
 "kind": "named|generic", "canonical_name": "standard name"}
Rules:
- keep=false for extraction errors, fragments, numbers, dates, news outlets or bylines, and vague phrases ("the company", "experts", "users").
- type: company = business; technology = AI model, product, platform or technology field; government_institution = governments, countries, regulators, public agencies, militaries; person = a named individual; other_organization = universities, NGOs, standards bodies, research labs not run as companies.
- kind=generic for technology fields or categories ("machine learning", "large language models", "chatbots"); named for specific entities or products.
- canonical_name: the standard written form, e.g. "OpenAI", "Nvidia", "GPT-4", "United States". Merge obvious variants to the parent name only when they refer to the same thing (e.g. "Facebook" -> "Meta"), not products to their makers."""


def validate_review(d: dict) -> dict:
    t = d.get("type")
    if t not in MODEL_TYPES + ["other_organization"]:
        raise ValueError(f"bad type {t}")
    name = normalize_spaces(d.get("canonical_name", ""))
    if d.get("keep") and not name:
        raise ValueError("missing name")
    return {"keep": bool(d.get("keep")), "type": t, "kind": "generic" if d.get("kind") == "generic" else "named",
            "canonical_name": name[:80]}


def main(limit: int | None = None) -> None:
    keep_blocks = pd.read_parquet(config.out("corpus", "blocks.parquet"), columns=["block_uid", "keep"])
    keep_blocks = set(keep_blocks.loc[keep_blocks["keep"], "block_uid"])

    cols = ["doc_id", "block_id", "block_uid", "date", "domain", "entity_surface", "raw_label", "raw_score",
            "char_start", "char_end"]
    with timed(log, "load raw mentions"):
        m = pd.read_parquet(config.RAW_MENTIONS, columns=cols)
    n_raw = len(m)
    log.info("raw mentions: %d, min score %.4f", n_raw, m["raw_score"].min())
    m = m[m["block_uid"].isin(keep_blocks) & (m["raw_score"] >= MIN_SCORE)]
    m = m[m["raw_label"].isin(LABEL_TO_TYPE.keys())].copy()
    m["surface"] = m["entity_surface"].map(normalize_spaces)
    m = m[(m["surface"].str.len() >= 2) & ~m["surface"].str.match(NUMERIC_JUNK_RE)]
    m["type_raw"] = m["raw_label"].map(LABEL_TO_TYPE)

    with timed(log, "canonical keys"):
        uniq = pd.Series(m["surface"].unique())
        key_of = dict(zip(uniq, uniq.map(entity_key)))
        m["key"] = m["surface"].map(key_of)
        m["key"] = m["key"].map(lambda k: entity_key(ALIAS_MAP[k]) if k in ALIAS_MAP else k)
    m = m[m["key"].str.len() >= 2]
    log.info("mentions after filters: %d (%.1f%% of raw)", len(m), 100 * len(m) / n_raw)

    # Display name: alias target if any, else the most frequent surface form.
    surf = m.groupby(["key", "surface"]).size().rename("n").reset_index()
    surf = surf.sort_values(["key", "n"], ascending=[True, False]).drop_duplicates("key")
    name = dict(zip(surf["key"], surf["surface"]))
    for k, n in ALIAS_MAP.items():
        name[entity_key(n)] = n
    tv = m.groupby(["key", "type_raw"])["raw_score"].sum().rename("w").reset_index()
    tv = tv.sort_values(["key", "w"], ascending=[True, False]).drop_duplicates("key")
    ents = m.groupby("key").agg(n_mentions=("doc_id", "size"), n_docs=("doc_id", "nunique"),
                                n_domains=("domain", "nunique"), mean_score=("raw_score", "mean")).reset_index()
    ents["name_rule"] = ents["key"].map(name)
    ents["type_rule"] = ents["key"].map(dict(zip(tv["key"], tv["type_raw"])))
    ents = ents.sort_values("n_docs", ascending=False).reset_index(drop=True)
    log.info("canonical keys: %d; with >=30 docs: %d", len(ents), int((ents["n_docs"] >= 30).sum()))

    # LLM review of the head, with two short contexts per entity.
    head = ents.head(REVIEW_TOP_N)
    ctx = (m[m["key"].isin(set(head["key"]))]
           .sort_values("raw_score", ascending=False)
           .drop_duplicates(["key", "doc_id"])
           .groupby("key").head(2))
    blocks_txt = pd.read_parquet(config.out("corpus", "blocks.parquet"), columns=["block_uid", "text"])
    blocks_txt = blocks_txt[blocks_txt["block_uid"].isin(set(ctx["block_uid"]))].set_index("block_uid")["text"]
    snippets: dict[str, list[str]] = {}
    for r in ctx.itertuples():
        t = blocks_txt.get(r.block_uid, "")
        s = max(0, int(r.char_start) - 120)
        snippets.setdefault(r.key, []).append(truncate(t[s:int(r.char_end) + 120], 280))
    top_surfaces = (surf.set_index("key")["surface"]).to_dict()
    variants = m[m["key"].isin(set(head["key"]))].groupby(["key", "surface"]).size().rename("n").reset_index() \
        .sort_values("n", ascending=False).groupby("key")["surface"].apply(lambda s: list(s[:3]))

    jobs = []
    for r in head.itertuples():
        user = (f"Candidate: {r.name_rule}\nSurface forms: {', '.join(variants.get(r.key, [top_surfaces.get(r.key, r.name_rule)]))}\n"
                f"Extractor type: {r.type_rule}\nDocuments: {r.n_docs}\nContexts:\n"
                + "\n".join(f"- {s}" for s in snippets.get(r.key, [])))
        jobs.append(llm.Job(id=r.key, system=REVIEW_SYSTEM, user=user))
    review_path = config.out("entities", "entity_review_llm.jsonl")
    llm.run_batch("entity_review", jobs, review_path, validate_review, max_tokens=90, limit=limit)

    rev = pd.DataFrame([{"key": r["id"], **r["label"]} for r in llm.read_results(review_path)])
    ents = ents.merge(rev, on="key", how="left")
    ents["reviewed"] = ents["keep"].notna()
    ents["keep"] = ents["keep"].astype("boolean").fillna(True).astype(bool)
    ents["type"] = ents["type"].fillna(ents["type_rule"])
    ents["kind"] = ents["kind"].fillna("named")
    ents["name"] = ents["canonical_name"].where(ents["canonical_name"].fillna("").str.len() > 0, ents["name_rule"])
    ents.loc[ents["type"] == "organization", "type"] = "other_organization"

    # Merge on the reviewed name: the entity id is the key of the reviewed name.
    ents["entity_id"] = ents["name"].map(entity_key)
    ents["entity_id"] = ents["entity_id"].map(lambda k: entity_key(ALIAS_MAP[k]) if k in ALIAS_MAP else k)
    key_to_id = dict(zip(ents["key"], ents["entity_id"]))
    kept_keys = set(ents.loc[ents["keep"], "key"])
    m = m[m["key"].isin(kept_keys)].copy()
    m["entity_id"] = m["key"].map(key_to_id)

    # Type and name per merged id come from its most-documented member key.
    rep = ents[ents["keep"]].sort_values("n_docs", ascending=False).drop_duplicates("entity_id")
    final = m.groupby("entity_id").agg(n_mentions=("doc_id", "size"), n_docs=("doc_id", "nunique"),
                                       n_domains=("domain", "nunique")).reset_index()
    final = final.merge(rep[["entity_id", "name", "type", "kind", "reviewed"]], on="entity_id", how="left")
    final["n_keys_merged"] = final["entity_id"].map(ents[ents["keep"]].groupby("entity_id").size())
    final = final.sort_values("n_docs", ascending=False).reset_index(drop=True)
    final.to_parquet(config.out("entities", "entities.parquet"), index=False)

    m = m.drop(columns=["entity_surface", "raw_label"])
    m.to_parquet(config.out("entities", "mentions.parquet"), index=False)

    rv = ents[ents["reviewed"]]
    stats = {
        "min_score": MIN_SCORE, "raw_mentions": n_raw, "mentions_kept": len(m),
        "keys_before_review": len(ents), "reviewed": int(len(rv)),
        "review_dropped": int((~rv["keep"]).sum()),
        "review_retyped": int((rv["type"] != rv["type_rule"]).sum()),
        "final_entities": len(final),
        "final_entities_ge30_docs": int((final["n_docs"] >= 30).sum()),
        "type_counts_ge30_docs": final[final["n_docs"] >= 30]["type"].value_counts().to_dict(),
        "top30": final.head(30)[["name", "type", "kind", "n_docs"]].to_dict("records"),
    }
    write_json(stats, config.out("entities", "entity_stats.json"))
    log.info("final entities %d (%d with >=30 docs)", len(final), stats["final_entities_ge30_docs"])


if __name__ == "__main__":
    import sys
    lim = next((int(a.split("=")[1]) for a in sys.argv if a.startswith("--limit=")), None)
    main(limit=lim)
