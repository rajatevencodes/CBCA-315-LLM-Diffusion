# 03. Plan and Project Structure

```text
CBCA315-Semantic-Diffusion/
├── README.md                  what the project is + quick start + where each submission item is
├── setup.md                   step-by-step guide, from zero
├── Makefile                   one command per step (make help)
├── pyproject.toml, uv.lock    uv project (Python 3.13)
├── requirements.txt           for Google Colab (make requirements)
├── .env.example               copy to .env and add OPENROUTER_API_KEY
├── data/
│   ├── raw/                   cora_orig.zip, PubMed_orig.zip (TAPE release, checksummed)
│   └── processed/             aligned nodes/edges CSVs, embeddings, edge cosines
├── src/
│   ├── config.py              all settings
│   ├── load_data.py           step 1: download, verify, align text by ID, synthetic graph
│   ├── edge_weights.py        step 2: embeddings, W0/W1/W2, edge selection, all variants
│   ├── llm_scoring.py         step 3: OpenRouter client, cache, cost ledger, spending ceiling
│   ├── detect.py              step 4: LPA / Leiden / Infomap + metrics -> runs.csv
│   ├── diffusion.py           step 5: Independent Cascade downstream task
│   ├── make_charts.py         step 6: tables, significance tests, charts
│   └── run_pipeline.py        runs steps 1-6
├── notebooks/                 Colab notebook (data → method → results)
├── outputs/                   runs.csv, diffusion.csv, significance.csv, summary.json,
│                              llm/ (calls.jsonl, scores/), tables/*.md, charts/*.png
├── reports/                   00_front_matter.md … 05_chapter5.md, appendix_a.md, presentation.md, assets/
└── docs/                      cbca-315-doc.md (requirements), 01 understanding, 02 decisions, 03 plan
```

| Step | Output | Status |
|---|---|---|
| 1 | Requirements merged, datasets downloaded and aligned, uv project | Done |
| 2 | Pipeline code (`src/`) | Done |
| 3 | LLM scoring, experiments, downstream task, tables, charts (`outputs/`) | See README "Results at a glance" |
| 4 | Report chapters + presentation (`reports/`) | Draft done |
| 5 | GitHub repo, Colab link, Overleaf (journal template), tag `v1.0-results-freeze` | You |
| 6 | Instructor approval, similarity report, journal submission, viva | You |
