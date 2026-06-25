# An intent-aware scientific paper retrieval and recommendation system based on fine-grained semantic representation

This directory contains a cleaned and publish-ready standalone release for three core stages:

1. Data preprocessing from arXiv ID + LaTeX source to extracted paper JSON.
2. Semantic unit generation from extracted paper data.
3. Retrieval service (hybrid-doc-semantic), reranking, and knowledge output (brief answer + recommendation reasons).

The implementation focuses on:

- Stable and explicit interfaces.
- Batch-friendly orchestration with resume support.
- Clear CLI entry points.
- Fully standalone code in `src/scientific_resource_release`.

## Directory Layout

```text
release/
  pyproject.toml
  README.md
  src/scientific_resource_release/
    cli.py
    models.py
    preprocess/pipeline.py
    semantic_units/pipeline.py
    retrieval/
      database.py
      schemas.py
      ingestion.py
      algorithms.py
      service.py
    utils/
      io.py
      logging_utils.py
      paths.py
      semantic_scholar.py
```

## Install

From `ScientificResource/release`:

```bash
pip install -e .
```

## Quick Start

### 1) Preprocess a single paper

```bash
sr-release preprocess one \
  --arxiv-id 2401.00663 \
  --project-root ./ScientificResource
```

### 2) Preprocess in batch (from JSONL)

```bash
sr-release preprocess batch \
  --project-root ./ScientificResource \
  --jsonl ./ScientificResource/data/raw/arxiv_ai_2024_2025_augmented.jsonl
```

### 3) Generate semantic units for one paper

```bash
sr-release semantic-units one \
  --arxiv-id 2401.00663 \
  --project-root ./ScientificResource
```

### 4) Generate semantic units in batch

```bash
sr-release semantic-units batch \
  --project-root ./ScientificResource
```

### 5) Ingest extracted results into PostgreSQL + pgvector

```bash
sr-release retrieval ingest \
  --data-root ./ScientificResource/data/extracted
```

### 6) Run hybrid-doc-semantic retrieval

```bash
sr-release retrieval search \
  --query "graph rag for scientific qa" \
  --top-k 10 \
  --intent-decomposer llm
```

### 7) Run retrieval + rerank + knowledge outputs

```bash
sr-release retrieval search \
  --query "how to improve multimodal retrieval grounding" \
  --top-k 10 \
  --with-knowledge \
  --reason-top-k 5
```

## Notes

- This release writes outputs to `ScientificResource/data/extracted/{arxiv_id}` by default.
- Semantic Scholar metadata refresh is supported in preprocessing and is non-blocking.
- Retrieval uses two stages: doc-level hybrid retrieval plus semantic-representation dense matching, then intent-channel weighted RRF fusion.
- Knowledge mode adds cross-encoder rerank, recommendation reason generation, and a concise grounded answer.
- LLM-related capability is optional and controlled by env vars or CLI args:
  - `LLM_API_KEY`
  - `LLM_BASE_URL` (default: `https://api.openai.com/v1`)
  - `LLM_MODEL` (default: `gpt-4o-mini`)
- Set retrieval/reranker/database env values by copying `release/env.sample`.
