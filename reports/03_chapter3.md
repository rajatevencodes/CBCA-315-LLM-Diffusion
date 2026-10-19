# Chapter 3. Project implementation

The Chapter 2 design is implemented in `src/`. Reference numbers use the consolidated list in Section 5.6.

## 3.1 Development environment

| Item | Version / detail | Role |
|---|---|---|
| Computer | Apple Silicon Mac; experiments ran in a Linux (aarch64) virtual machine with 4 vCPUs and 4 GB RAM, CPU only | Development and all runs |
| Python | 3.13.15 | Programming language |
| uv | 0.12 | Creates the environment from `uv.lock` |
| NetworkX | 3.7 | Graph building, label propagation [25] |
| python-igraph | 1.0.0 | Leiden, Infomap, modularity [24] |
| sentence-transformers / PyTorch (CPU) | 5.7.0 / 2.14.1 | MiniLM embeddings [6], [22] |
| NumPy, pandas, SciPy, scikit-learn | 2.5.3, 3.0.6, 1.18.1, 1.9.1 | Arrays, tables, Hungarian matching and Wilcoxon test, NMI / ARI / F1 |
| Matplotlib | 3.11.2 | Charts |
| requests, python-dotenv | 2.34.2, 1.x | OpenRouter API calls; reads the key from `.env` |
| LLM service | OpenRouter, `meta-llama/llama-3.1-8b-instruct` on DeepInfra (fp8) | Edge scoring [23] |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2`, revision `1110a243` | Frozen text encoder |

This is a uv project. `pyproject.toml` lists the libraries, `uv.lock` fixes every version, and `.python-version` selects Python 3.13. On Linux, PyTorch comes from the CPU-only index so no GPU packages are downloaded. The `Makefile` wraps every step:

```bash
make init          # install Python 3.13 + packages, create .env for the API key
make all           # run steps 1-6 (data → weights → llm → experiments → diffusion → charts)
make help          # list every command
```

## 3.2 Project structure

```text
CBCA315-Semantic-Diffusion/
├── data/raw/            cora_orig.zip, PubMed_orig.zip (downloaded, checksummed, never changed)
├── data/processed/      aligned nodes/edges CSVs, embeddings, edge cosines
├── src/                 the 8 Python modules
├── notebooks/           semantic_diffusion.ipynb (Google Colab version of the pipeline)
├── outputs/             runs.csv, diffusion.csv, significance.csv, summary.json,
│                        llm/calls.jsonl, llm/scores/, tables/, charts/
└── reports/             this report
```

## 3.3 Step 1: loading and aligning the data (`load_data.py`)

`download()` fetches each archive from the Google Drive link in the TAPE README and stops if its SHA-256 differs from the value stored in `config.py`. This guarantees everyone uses the same files.

`parse_cora()` reads `cora.content` (IDs, classes) and `cora.cites` (citations). Text is found through the `papers` index file. 945 Cora IDs point to more than one extraction file, so the function picks the file with the most complete title and abstract. It never reuses another paper's text:

```python
for pid in ids:
    ti, ab = max((title_abstract(f) for f in files[pid]),
                 key=lambda t: (bool(t[0]) + bool(t[1]), len(t[1])))
    texts.append(f"Title: {ti}\nAbstract: {ab}")
```

`parse_pubmed()` builds a dictionary from PMID to text and keeps only nodes whose PMID has text. Joining by key instead of by row position is the fix described in Section 2.4:

```python
text_by_id = {str(r["PMID"]): f"Title: {r.get('TI') or ''}\nAbstract: {r.get('AB') or ''}"
              for r in records if r.get("PMID") and (r.get("TI") or r.get("AB"))}
keep = [pid for pid in all_ids if pid in text_by_id]
```

`undirected()` removes self-loops and duplicates and stores each edge once as (u, v) with u < v. The row number of an edge is its permanent edge ID, used by every later step.

`synthetic()` builds the 4-community stochastic block model and gives each node 12 words drawn from its community's topic list (guide Section 7.3).

## 3.4 Step 2: edge weights (`edge_weights.py`)

`embed()` encodes texts in chunks of 2,048 and saves each chunk, so a stopped run resumes where it left off. Embeddings are L2-normalised, so the cosine of an edge is a dot product:

```python
cos = np.einsum("ij,ij->i", emb[g.edges[:, 0]], emb[g.edges[:, 1]])
```

`select_band()` implements line 4-5 of Algorithm 1:

```python
inside = np.flatnonzero((cos >= lo) & (cos <= hi))
order = np.lexsort((inside, np.abs(cos[inside] - (lo + hi) / 2)))   # midpoint first, then edge id
return inside[order][:k]
```

`variants()` is the single list of every experiment variant (W0, W1, W2, random, bands, prompt, 70B, budgets). Both the LLM step and the detection step read it, so they always agree on which edges belong to which variant. `weights_for()` turns a variant into a weight vector; a W2 edge whose LLM answer failed keeps its W1 weight.

## 3.5 Step 3: LLM scoring (`llm_scoring.py`)

The client is about 100 lines and uses `requests` directly. The request pins the provider and turns off fallback, so the model being evaluated cannot silently change:

```python
body = {"model": self.model, "messages": [{"role": "user", "content": text}],
        "temperature": 0, "max_tokens": 8,
        "provider": {"order": [self.provider], "allow_fallbacks": False, "require_parameters": True},
        "usage": {"include": True}}
```

`parse_score()` accepts only a whole number 0-10, optionally followed by a full stop. Each request is identified by a SHA-256 of model, provider, prompt name and the full prompt text. Every attempt, successful or not, is appended to `outputs/llm/calls.jsonl` with the answer, score, token counts, cost, latency, generation ID and the serving provider. Before every call the client checks the running total against the $1.00 ceiling. Eight requests run in parallel through a thread pool.

## 3.6 Step 4: detection and evaluation (`detect.py`)

`detect()` runs one detector with one seed. Python's `random` is seeded before igraph is called, because python-igraph draws its random numbers from it.

`cluster_acc()` builds the community × class contingency matrix and solves the one-to-one assignment with `scipy.optimize.linear_sum_assignment`. `evaluate()` adds NMI (arithmetic), ARI and modularity on the original unweighted graph.

`run()` writes one row per run to `outputs/runs.csv` immediately after the run, and skips any (dataset, variant, detector, seed) already in the file. An interrupted experiment therefore continues where it stopped. A row stores the suite, variant, detector, seed, the six metrics, detector time, peak process memory and the number of LLM-scored edges.

## 3.7 Step 5: downstream task (`diffusion.py`)

`live_edge_samples()` draws 200 random live-edge graphs (each edge kept with p = 0.05) and stores their connected components once. `spread()` counts the nodes in the components that contain a seed, which is exactly the Independent Cascade outcome. `community_seeds()` follows the guide's code (seeds in proportion to community size, highest degree first) and then fills any remaining slots with top-degree nodes, so every strategy has exactly k distinct seeds. A Wilcoxon test over the 200 paired simulations compares W2 and W1 seeds.

## 3.8 Step 6: tables and charts (`make_charts.py`)

This step reads only files in `outputs/` and writes paste-ready Markdown tables (`outputs/tables/t*.md`), `significance.csv`, `summary.json` and five charts. No number in Chapter 4 is typed by hand.

## 3.9 Running the project

| Command | What happens | Time on the test machine |
|---|---|---|
| `make data` | Download (about 820 MB once), verify, align | about 15 s after download |
| `make weights` | Embeddings + cosines | Cora 63 s, PubMed 947 s of encoding (once) |
| `make pilot` | 50 Cora edges through the LLM | see Section 4.5 |
| `make llm` | All selected edges for every variant | see Section 4.5 |
| `make experiments` | All detector runs | see Table 4.7 |
| `make diffusion` | Downstream task | under a minute |
| `make charts` | Tables, tests, charts | a few seconds |

## 3.10 Google Colab

`notebooks/semantic_diffusion.ipynb` runs the same six steps from a Drive copy of the project. It installs `requirements.txt` (exported from `uv.lock` without PyTorch, which Colab already has) and reads the API key from Colab Secrets. Because LLM answers are cached in `outputs/llm/calls.jsonl`, the notebook reproduces the results without new API spending.
