# Setup Guide (Step by Step)

This guide assumes you have **never run this project before**. Follow the steps in order. Each step says **what to do**, **why**, and **what you should see**.

**Time needed:** about 45 minutes the first time (downloads + embeddings). Afterwards everything is cached.
**Computer:** a Mac. On Windows, use the same commands in PowerShell (install `make` or run the `uv run` lines from the Makefile directly).

## The big picture (read this once)

```text
data/raw/*.zip → align text → MiniLM cosine (W1) → LLM re-scores 10% of edges (W2)
              → LPA / Leiden / Infomap × 5 seeds → influence spread → tables + charts
```

* **Python** does all the work. Settings are in `src/config.py`.
* **The LLM** is only called for the most ambiguous edges. Every answer is saved in `outputs/llm/calls.jsonl`, so you never pay twice.
* **Every number** in the report comes from a file in `outputs/`.

## Step 1: Install uv (the Python manager)

**Why:** `uv` installs Python 3.13 and the exact package versions, so everybody gets the same results.

```bash
brew install uv                                       # if you use Homebrew
curl -LsSf https://astral.sh/uv/install.sh | sh       # otherwise
uv --version
```

**You should see:** `uv 0.x.x`.

## Step 2: Go to the project folder and install

```bash
cd ~/Desktop/CBCA315-Semantic-Diffusion
make init
```

**Why:** creates `.venv` with Python 3.13 and all packages, and creates a `.env` file for your API key.

**You should see:** a list of installed packages, then `Created .env - paste your OPENROUTER_API_KEY into it`.

## Step 3: Add your OpenRouter key

1. Get a key at https://openrouter.ai/keys (add a few dollars of credit; the whole project costs well under $1).
2. Open `.env` in any editor and write:

```text
OPENROUTER_API_KEY=sk-or-v1-...
```

**Why:** the LLM step needs it. `.env` is in `.gitignore`, so the key never reaches GitHub.

## Step 4: Run the pipeline

Run the steps one at a time the first time, so you can see what each does:

```bash
make data          # downloads ~820 MB once, checks checksums, aligns text to nodes
make weights       # text embeddings (PubMed takes ~10-15 min on a laptop CPU, once)
make pilot         # 50 test LLM calls: check the answers look sensible (< $0.01)
make llm           # all LLM calls (~ 15-20k, cached, stops at $1.00)
make experiments   # all community-detection runs -> outputs/runs.csv
make diffusion     # downstream influence-spread task
make charts        # tables, significance tests, charts -> outputs/
```

or everything at once: `make all`.

**You should see:** each step prints a short summary. If a step stops (closed laptop, network), just run it again: finished work is skipped.

## Step 5: Look at the results

* `outputs/tables/*.md`: the report tables (open in VS Code preview)
* `outputs/charts/*.png`: the report figures
* `outputs/summary.json`: headline numbers
* `outputs/runs.csv`: one row per detector run (open in Excel)

## Step 6: Google Colab (required by the course)

1. Upload the whole project folder to Google Drive as `MyDrive/CBCA315-Semantic-Diffusion`.
2. Open `notebooks/semantic_diffusion.ipynb` in Colab.
3. Add the key under the 🔑 Secrets icon as `OPENROUTER_API_KEY`.
4. Runtime → Run all. Cached LLM answers are reused, so it costs nothing.

## Step 7: GitHub (required by the course)

```bash
git add .
git commit -m "Pipeline, results and report"
git remote add origin https://github.com/<your-username>/CBCA315-Semantic-Diffusion.git
git push -u origin main
git tag v1.0-results-freeze && git push --tags     # at CP2 (Thu 15 Oct)
```

`data/raw/`, `data/processed/` and `.env` are ignored by Git. `outputs/` (including the LLM cache) **is** committed: it is the evidence for every number.

## Troubleshooting

| Problem | Fix |
|---|---|
| `OPENROUTER_API_KEY is missing` | Step 3 |
| `HTTP 402` | Add credit on openrouter.ai |
| `HTTP 404 / no endpoints` | The pinned provider is down; change `LLM["provider"]` in `src/config.py` (and report it) |
| `checksum mismatch` | Delete the file in `data/raw/` and run `make data` again |
| `... missing: run make llm first` | Run `make llm` before `make experiments` |
| `STOPPED, spending ceiling reached` | Raise `SPEND_CEILING_USD` in `src/config.py` only if you mean to |
