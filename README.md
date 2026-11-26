# Semantic Diffusion: Budgeted LLM Edge Weights for Community Detection

**Course:** CBCA315 Minor Project, Topic 2: Information Diffusion Using LLMs for Community Detection · BCA, Bennett University
**Student:** Rajat Sharma (S24BCAU0134) · **Guide:** Dr. Soumita Das

Each paper is a node and each citation an edge. We weight every edge by how similar the two papers' texts are, ask a small LLM (Llama 3.1 8B) to re-judge only the most ambiguous **10%** of edges, and measure whether communities found by label propagation, Leiden and Infomap improve enough to justify the LLM's cost.

```text
Raw archives → [1] Align text → [2] MiniLM cosine (W1) → [3] LLM re-scores ambiguous edges (W2)
            → [4] LPA / Leiden / Infomap × 5 seeds → [5] Influence spread → [6] Tables + charts
```

## Results at a glance

<!-- RESULTS -->
Mean NMI over 5 seeds (full table with ARI, ACC, F1 and modularity: `outputs/tables/t2_main_results.md`).

| Dataset | Detector | W0 structure | W1 cosine | W2 cosine + LLM |
|---|---|---|---|---|
| Cora | LPA | 0.406 | 0.406 | 0.406 |
| Cora | Leiden | 0.466 | 0.454 | 0.449 |
| Cora | Infomap | 0.413 | 0.417 | 0.413 |
| PubMed | LPA | 0.180 | 0.184 | 0.184 |
| PubMed | Leiden | 0.208 | 0.207 | 0.204 |
| PubMed | Infomap | 0.172 | 0.175 | 0.172 |

* W2 did not beat W1. Over 30 paired runs W2 won 7 and lost 23 (mean NMI change −0.0026, Wilcoxon p = 0.0008).
* The 8B model gave most edges the same score (8), so it added little information. The 70B model, other prompts and other bands did not change this (`outputs/tables/t4_ablation_nmi.md`).
* Synthetic sanity check: W1 and W2 recover the 4 planted communities (NMI 0.99 to 1.00).
* Total LLM spend for the whole project: $0.2151 (`outputs/llm/calls.jsonl`).
<!-- /RESULTS -->

## Quick start

```bash
make init        # Python 3.13 + packages, creates .env (paste OPENROUTER_API_KEY)
make all         # every step; finished work is cached and skipped on re-runs
make help        # list all commands
```

New to this? Follow **[setup.md](setup.md)** step by step.

## Project structure

```text
├── README.md, setup.md        overview and step-by-step guide
├── Makefile                   one command per step
├── pyproject.toml, uv.lock    uv project (Python 3.13); requirements.txt for Colab
├── data/raw/                  cora_orig.zip, PubMed_orig.zip (TAPE release, checksummed)
├── data/processed/            aligned nodes/edges, embeddings, edge cosines
├── src/
│   ├── config.py              all settings
│   ├── load_data.py           step 1: download, verify, align text by paper ID
│   ├── edge_weights.py        step 2: embeddings, W0/W1/W2, edge selection
│   ├── llm_scoring.py         step 3: OpenRouter client, cache, cost ledger, $1 ceiling
│   ├── detect.py              step 4: LPA / Leiden / Infomap + metrics
│   ├── diffusion.py           step 5: Independent Cascade downstream task
│   ├── make_charts.py         step 6: tables, significance tests, charts
│   └── run_pipeline.py        runs steps 1-6
├── notebooks/                 semantic_diffusion.ipynb (Google Colab)
├── outputs/                   runs.csv, diffusion.csv, significance.csv, summary.json,
│                              llm/ (calls.jsonl, scores/), tables/, charts/
├── reports/                   report chapters (Markdown) and presentation content
└── docs/                      requirements and design notes
```

## Where each submission item is

| Requirement (course guide) | File(s) |
|---|---|
| CP1: literature matrix, gap, research question, objectives | `reports/01_chapter1.md` |
| CP1: base paper, datasets, metrics, pseudocode, baselines, protocol | `reports/02_chapter2.md` |
| CP2: code running end-to-end, sanity test, main table, ablation, sensitivity, runtime, downstream, significance | `src/`, `outputs/`, `reports/03_chapter3.md`, `reports/04_chapter4.md` |
| Paper content (journal draft source) | `reports/00_front_matter.md` … `05_chapter5.md`, `appendix_a.md` |
| Viva presentation content | `reports/presentation.md` |
| Colab notebook | `notebooks/semantic_diffusion.ipynb` |
| Requirements (merged course guide) | `docs/cbca-315-doc.md` |
| Design decisions | `docs/02_decisions_log.md` |

## Data

Cora and PubMed with original titles and abstracts, from the release linked in the TAPE repository (He et al., ICLR 2024, https://github.com/XiaoxinHe/TAPE). `make data` downloads them and checks their SHA-256. Raw data is not committed to Git.
