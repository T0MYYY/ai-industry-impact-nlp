# AI Industry Impact Analysis via NLP — UChicago ADSP 32018 Final Project

<!-- BADGES_BEGIN -->
<p align="center">
  <img alt="Course" src="https://img.shields.io/badge/Course-ADSP%2032018-DC143C?style=flat-square&labelColor=2a323d">
  <img alt="UChicago" src="https://img.shields.io/badge/UChicago-Next--Gen%20NLP-800000?style=flat-square&labelColor=2a323d">
  <img alt="Term" src="https://img.shields.io/badge/Term-Winter%202025-2a323d?style=flat-square&labelColor=2a323d">
  <img alt="Author" src="https://img.shields.io/badge/Author-Solo-1f7a3d?style=flat-square&labelColor=2a323d">
  <img alt="Status" src="https://img.shields.io/badge/Status-Final-ec5800?style=flat-square&labelColor=2a323d">
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&labelColor=2a323d&logo=python&logoColor=white">
  <img alt="Jupyter" src="https://img.shields.io/badge/Jupyter-notebook-F37626?style=flat-square&labelColor=2a323d&logo=jupyter&logoColor=white">
  <img alt="🤗 Transformers" src="https://img.shields.io/badge/🤗%20Transformers-5.0-FFD21E?style=flat-square&labelColor=2a323d">
  <img alt="sentence-transformers" src="https://img.shields.io/badge/sentence--transformers-3.3-FFB000?style=flat-square&labelColor=2a323d">
  <img alt="spaCy" src="https://img.shields.io/badge/spaCy-3.8-09A3D5?style=flat-square&labelColor=2a323d&logo=spacy&logoColor=white">
  <img alt="BERTopic" src="https://img.shields.io/badge/BERTopic-0.16-7B68EE?style=flat-square&labelColor=2a323d">
  <img alt="UMAP" src="https://img.shields.io/badge/UMAP-0.5-A38EE6?style=flat-square&labelColor=2a323d">
  <img alt="HDBSCAN" src="https://img.shields.io/badge/HDBSCAN-0.8-8B5CF6?style=flat-square&labelColor=2a323d">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.10-EE4C2C?style=flat-square&labelColor=2a323d&logo=pytorch&logoColor=white">
  <img alt="GLiNER" src="https://img.shields.io/badge/GLiNER-0.2-1F2937?style=flat-square&labelColor=2a323d">
  <img alt="pandas" src="https://img.shields.io/badge/pandas-2.2-150458?style=flat-square&labelColor=2a323d&logo=pandas&logoColor=white">
</p>
<!-- BADGES_END -->

**Course:** ADSP 32018 — Next-Gen NLP: Transformers, LLMs and Agentic AI in Practice (Winter 2025, University of Chicago)  
**Author:** Chiyang Chen  
**Dataset:** ~200K news articles on AI, ML, and data science

---

## Project Overview

An end-to-end NLP pipeline that mines ~200K tech news articles to answer a core question: **which industries will be most impacted by AI, how (positively or negatively), and through what mechanisms?**

Motivated by the 2023 Goldman Sachs report estimating ~25% of US/Europe tasks are automatable by AI, and validated by Facebook Research's Moravec's Paradox findings — that AI disrupts cognitive/office work far more than physical/sensorimotor tasks.

**Research Questions**

1. Which industries and companies are most likely to be impacted by AI over the next several years?
2. How will they be impacted — positively, negatively, or ambiguously — and through what means (automation, augmentation, cost reduction, workflow redesign)?
3. What factors make AI adoption successful or unsuccessful?

---

## Pipeline Overview

```mermaid
flowchart TD
    A["~200K News Articles<br/>(Parquet)"] --> B["01 EDA & Data Profiling"]

    B --> C["02A Sample & Label Blocks"]
    B --> D["02C Sample & Label Sentences"]

    C --> E["02B Block-level AI Relevance Classifier<br/>→ Filter Corpus"]
    D --> F["02D Sentence-level Classifier<br/>→ Rebuild Clean Docs"]

    E --> G["AI-Relevant Article Corpus"]
    F --> G

    G --> H["03 Topic Modeling<br/>(BERTopic)"]
    G --> I["04 Entity Extraction<br/>(GLiNER + LLM)"]
    G --> J["05A/B/C Sentiment Analysis<br/>(custom-trained model)"]

    H --> K["06 Presentation Assets<br/>(industry dashboards,<br/>sentiment over time,<br/>entity impact maps)"]
    I --> K
    J --> K
```

---

## Notebooks

| Notebook | Stage | Description |
|---|---|---|
| `01_data_ingestion_eda.ipynb` | EDA | Load parquet corpus; profile shape, fields, missing values, time distribution, and duplicates |
| `02A_block_sample_and_label.ipynb` | Filtering | Sample article text blocks; manually label for AI-relevance |
| `02B_block_train_predict.ipynb` | Filtering | Train block-level classifier; threshold selection; predict over full corpus to isolate AI-relevant blocks |
| `02C_sentence_sample_and_label.ipynb` | Filtering | Sample and label at sentence granularity for finer-grained filtering |
| `02D_sentence_train_predict_and_rebuild.ipynb` | Filtering | Train sentence classifier; predict; reconstruct cleaned AI-relevant document corpus |
| `03_topic_modeling.ipynb` | Analysis | BERTopic on filtered corpus; identify industry themes and AI application clusters |
| `04_entity_extraction.ipynb` | Analysis | GLiNER NER to extract organizations and technologies; LLM-based canonical name cleaning |
| `05A_sentiment_dataset_creation.ipynb` | Sentiment | Construct labeled sentiment training dataset from news content |
| `05B_sentiment_model_training.ipynb` | Sentiment | Train custom sentiment classifier (fine-tuned; no pre-labeled HuggingFace models used) |
| `05C_sentiment_inference_and_aggregation.ipynb` | Sentiment | Run inference at scale; aggregate by topic, entity, and time |
| `06_presentation_assets.ipynb` | Output | Final visualizations: industry impact rankings, sentiment trends over time, entity-level breakdowns |

---

## Tech Stack

| Component | Tools |
|---|---|
| Data handling | `pandas`, `pyarrow` |
| Article filtering | Custom block & sentence classifiers (`scikit-learn`) |
| Topic modeling | `BERTopic`, `sentence-transformers`, `UMAP`, `HDBSCAN` |
| Named entity recognition | `GLiNER`, LLM API (canonical cleaning) |
| Sentiment analysis | Custom fine-tuned model (trained from labeled data) |
| Visualization | `matplotlib`, `plotly` |

---

## Key Findings

See the presentation slides and `06_presentation_assets.ipynb` for the full analysis. High-level results cover:
- Industries with highest AI exposure (legal, finance, healthcare, office automation)
- Company-level sentiment breakdown (who is positioned positively vs. at risk)
- Temporal trends in AI adoption sentiment (2020–2023)
- Technologies driving impact (LLMs, automation tools, robotics)

---

## How to Run

```bash
jupyter notebook 01_data_ingestion_eda.ipynb
```

Due to `ipywidgets` compatibility, some interactive outputs may not render in GitHub's notebook viewer. Clone the repo and run locally, or open in [nbviewer](https://nbviewer.org/).
