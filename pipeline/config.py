"""Paths and run-wide settings for the analysis pipeline."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_DIR = Path(os.environ.get("PROJECT_DIR", "/content/drive/MyDrive/NLP/NLP_FINAL_PROJECT_Tom_Chen"))
UPSTREAM_DIR = PROJECT_DIR / "output"
OUT_DIR = PROJECT_DIR / "results"

# Local scratch on the Colab VM; much faster than Drive for large intermediate files.
SCRATCH_DIR = Path(os.environ.get("SCRATCH_DIR", "/content/scratch"))

SEED = 42

# Upstream artifacts: 02D clean corpus, 05A GLiNER mentions, 06A/06B sentiment labels and model
CLEAN_BLOCKS = UPSTREAM_DIR / "02D_sentence_train" / "clean_ai_blocks.parquet"
CLEAN_DOCS = UPSTREAM_DIR / "02D_sentence_train" / "clean_ai_docs.parquet"
RAW_MENTIONS = UPSTREAM_DIR / "04_entity_extraction" / "04A_raw_mentions" / "entity_mentions_raw.parquet"
SENTIMENT_LABELS = UPSTREAM_DIR / "05_sentiment_dataset" / "sentiment_training_data.parquet"
SENTIMENT_MODEL = UPSTREAM_DIR / "05_sentiment_model" / "best_model"
SENTIMENT_VALID = UPSTREAM_DIR / "05_sentiment_model" / "valid_split.parquet"

# DeepSeek
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-flash"

# USD per 1M tokens, peak rates; off-peak is half (api-docs.deepseek.com/quick_start/pricing).
PRICE_PEAK = {"input_hit": 0.006, "input_miss": 0.30, "output": 1.20}
# Peak windows in UTC, Monday-Friday: 01:00-04:00 and 06:00-10:00.
PEAK_WINDOWS_UTC = [(1, 4), (6, 10)]

# Spend caps in USD; the ledger enforces them across restarts.
BUDGET_TOTAL_USD = 8.0
BUDGET_TASK_USD = {
    "topic_labels": 0.2,
    "entity_review": 2.0,
    "sentiment_test": 0.6,
    "doc_labels": 4.5,
}


def out(*parts: str) -> Path:
    p = OUT_DIR.joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def scratch(*parts: str) -> Path:
    p = SCRATCH_DIR.joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
