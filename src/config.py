"""All settings in one place.

To change a path, a model or an experiment setting, change it HERE.
Every other file imports its settings from this module.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Folders
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"              # downloaded archives (never changed)
PROCESSED_DIR = ROOT / "data" / "processed"  # aligned nodes/edges, embeddings
OUTPUT_DIR = ROOT / "outputs"                # every number in the report comes from here
LLM_DIR = OUTPUT_DIR / "llm"                 # LLM ledger (calls.jsonl) and scores/
CHARTS_DIR = OUTPUT_DIR / "charts"
TABLES_DIR = OUTPUT_DIR / "tables"

# ---------------------------------------------------------------------------
# Data: original-text releases linked from the TAPE README (He et al., ICLR 2024)
# name -> (archive file, Google Drive id, expected SHA-256)
# ---------------------------------------------------------------------------
DATASETS = ["cora", "pubmed"]
SOURCES = {
    "cora": ("cora_orig.zip", "1hxE0OPR7VLEHesr48WisynuoNMhXJbpl",
             "4449b98350598f1e5ed1726923410adf116a6ac3c8fd5dcea40e67ad296869fa"),
    "pubmed": ("PubMed_orig.zip", "1sYZX-jP6H8OkopVa9cp8-KXdEti5ki_W",
               "dc562ea54854db62b9311e642ef1c1ba8fbe3d90c5e8e6afc596278fd53ba92f"),
}
# Synthetic sanity graph: 4 planted communities, each with its own topic words.
SYNTHETIC = {"blocks": 4, "block_size": 100, "p_in": 0.08, "p_out": 0.02, "words_per_node": 12, "seed": 7}

# ---------------------------------------------------------------------------
# Edge weights
# W0 = 1 for every edge, W1 = text cosine, W2 = W1 with selected edges re-scored by the LLM
# ---------------------------------------------------------------------------
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"  # pinned Hugging Face commit
WEIGHT_FLOOR = 1e-6       # keeps every weight strictly positive
BAND = (0.3, 0.6)         # "ambiguous" cosine band from the course guide
BUDGET = 0.10             # LLM re-scores at most 10% of all edges

# ---------------------------------------------------------------------------
# LLM (OpenRouter). Key comes from .env -> OPENROUTER_API_KEY
# ---------------------------------------------------------------------------
LLM = {
    "url": "https://openrouter.ai/api/v1/chat/completions",
    "model": "meta-llama/llama-3.1-8b-instruct",
    "provider": "deepinfra/fp8",   # pinned endpoint, no silent fallback to another provider
    "prompt": "influence",         # wording from the course guide, see llm_scoring.PROMPTS
    "temperature": 0,
    "max_tokens": 8,
    "max_chars": 600,              # characters of each paper sent to the LLM
    "workers": 8,                  # parallel requests
    "retries": 3,
    "timeout_s": 60,
}
SPEND_CEILING_USD = 1.00   # hard stop across every LLM call in the project
PILOT_EDGES = 50

# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------
SEEDS = [0, 1, 2, 3, 4]
DETECTORS = ["lpa", "leiden", "infomap"]
ABLATION_BANDS = [(0.2, 0.5), (0.4, 0.7)]   # where in the cosine range the LLM helps
ABLATION_DATASET = "cora"                   # prompt and model ablations run here only
ABLATION_PROMPTS = ["topic"]
ABLATION_MODELS = [("meta-llama/llama-3.1-70b-instruct", "amazon-bedrock", "70b")]
BUDGET_SWEEP = [0.05, 0.10, 0.20]

# Downstream task: Independent Cascade influence spread (course guide, Section 6.2)
DIFFUSION = {"detector": "leiden", "community_seed": 0, "k": [5, 10, 20, 50],
             "p": 0.05, "simulations": 200, "sim_seed": 0}
