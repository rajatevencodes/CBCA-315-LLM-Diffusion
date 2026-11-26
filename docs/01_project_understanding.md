# 01. Project Understanding

This is the **single source of truth**. The code, the outputs and the reports must all match this file.

**Title:** Budgeted LLM Refinement of Edge Weights for Community Detection in Citation Networks

**One line:** Each paper is a node and each citation an edge. We weight every edge by how similar the two papers' texts are, ask a small LLM to re-judge only the most ambiguous 10% of edges, and test whether communities found by diffusion-style detectors get better enough to justify the LLM's cost.

```text
Raw archives → [1] Align text to nodes → [2] MiniLM cosine (W1) → [3] LLM re-scores ambiguous edges (W2)
            → [4] LPA / Leiden / Infomap × 5 seeds → [5] Influence-spread task → [6] Tables + charts
```

## 1. Research question

> Does re-scoring a **limited** number of edge weights with a **small** LLM improve community detection in text-attributed citation networks enough to justify its API cost, compared with text-embedding weights alone?

Sub-questions:

1. Do text weights (W1) beat structure-only weights (W0)?
2. Does adding LLM scores to 10% of edges (W2) beat W1, and on which detector?
3. Does choosing ambiguous edges beat choosing the same number of random edges?
4. How do quality and cost change with the budget (5 / 10 / 20%), the prompt wording and the model size (8B vs 70B)?
5. Do better communities pick better seed nodes for spreading information (Independent Cascade)?

## 2. Research gap (from the course guide, Section 9, and the literature in Chapter 2)

* **Content-aware diffusion:** diffusion-based detectors (label propagation, Infomap) treat every edge as equally likely to carry information.
* **Cost vs benefit of LLMs:** LLM-based community papers (ComGPT, CommLLM) use large proprietary models and rarely report cost; whether an LLM beats cheap sentence embeddings is unclear.
* **Targeted LLM use:** querying the LLM only for uncertain edges under a fixed budget is largely unexplored for community detection. LLM-GNN (Chen et al., ICLR 2024) does budgeted LLM annotation, but for node classification.

We do **not** claim LLM edge weighting is new (the guide proposes it). The contribution is a measured quality / cost trade-off under an explicit budget, including a random-edge control.

## 3. Datasets (facts measured by `src/load_data.py`)

| Fact | Cora | PubMed |
|---|---|---|
| Source | TAPE release, `cora_orig.zip` | TAPE release, `PubMed_orig.zip` |
| Domain | Machine learning papers | Diabetes papers (PubMed) |
| Nodes used | 2,708 | 19,716 (1 paper without text excluded) |
| Undirected edges | 5,278 | 44,323 |
| Classes (evaluation only) | 7 | 3 |
| Text coverage | 2,373 title + abstract, 293 title only, 42 abstract only | all 19,716 title + abstract |

Plus a synthetic sanity graph: 4 planted communities × 100 nodes, p_in = 0.08, p_out = 0.02, each community with its own 3 topic words (course guide, Section 7.3).

## 4. Method

| Variant | Edge weight | LLM calls |
|---|---|---|
| W0 | 1 for every edge | 0 |
| W1 | cosine(MiniLM(text_u), MiniLM(text_v)), floored at 10⁻⁶ | 0 |
| W2 | W1, except the selected edges get LLM score / 10 | 10% of edges |

Edge selection for W2: edges with cosine in [0.3, 0.6], closest to 0.45 first, at most 10% of all edges (Cora 528, PubMed 4,432). Fixed before looking at any label.

LLM: Llama 3.1 8B Instruct via OpenRouter (DeepInfra fp8), temperature 0, max 8 tokens, first 600 characters of each paper, answer must be a single integer 0-10.

Detectors: weighted asynchronous label propagation (NetworkX), weighted Leiden with the modularity objective (igraph), weighted Infomap with 10 trials (igraph). Seeds 0-4.

## 5. Metrics

NMI (arithmetic), ARI, ACC with one-to-one Hungarian matching, macro-F1 over true classes, modularity Q on the original graph, number of communities, detector time, peak memory, LLM calls, tokens, cost (USD), latency. Paired Wilcoxon signed-rank tests with rank-biserial effect size.

## 6. Base paper and comparison (for the instructor)

The guide's Topic 2 plan (Section 9) is followed on the datasets it names. TAPE (He et al., ICLR 2024) provides the original text and dataset protocol; its supervised classification accuracy is not a clustering baseline, so no published numbers are imported. All baselines (W0 and W1 with LPA, Leiden, Infomap) are run by us under one protocol (guide Option A). ComGPT (IEEE TCSS 2026) and CommLLM (2025) are related work: ComGPT finds a local community around a seed node, CommLLM sends the whole graph to GPT-4o, so their numbers are not comparable. CDBNE (ESWA 2023) is the closest unsupervised fallback base paper if the instructor requires one.
