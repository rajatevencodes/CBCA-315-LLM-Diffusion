"""Step 2: text embeddings, edge weights, and which edges the LLM re-scores.

W0  every edge weight = 1                      (structure only)
W1  weight = cosine similarity of the two papers' MiniLM embeddings
W2  W1, but the selected edges get weight = LLM score / 10
"""
import json
import time

import numpy as np
import pandas as pd

import config
import load_data

SCORES_DIR = config.LLM_DIR / "scores"


# --------------------------------------------------------------------- embeddings
def embed(name, texts, chunk=2048):
    """Frozen MiniLM embeddings (L2-normalised). Saved in chunks so a stopped run resumes."""
    tag = config.EMBED_REVISION[:8]
    path = config.PROCESSED_DIR / f"{name}_emb_{tag}.npy"
    if path.exists():
        return np.load(path)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(config.EMBED_MODEL, revision=config.EMBED_REVISION, device="cpu")
    parts = []
    for i in range(0, len(texts), chunk):
        part = config.PROCESSED_DIR / f"{name}_emb_{tag}_part{i // chunk:02d}.npy"
        if not part.exists():
            t0 = time.time()
            np.save(part, model.encode(texts[i:i + chunk], batch_size=64, normalize_embeddings=True))
            part.with_suffix(".sec").write_text(f"{time.time() - t0:.2f}")
        parts.append(part)
    emb = np.concatenate([np.load(p) for p in parts])
    np.save(path, emb)
    seconds = sum(float(p.with_suffix(".sec").read_text()) for p in parts)
    path.with_suffix(".sec").write_text(f"{seconds:.2f}")
    for p in parts:  # chunks are no longer needed
        p.unlink()
        p.with_suffix(".sec").unlink()
    return emb


def cosine(name):
    """Cosine similarity for every edge (cached)."""
    path = config.PROCESSED_DIR / f"{name}_cos.npy"
    if path.exists():
        return np.load(path)
    g = load_data.load(name)
    emb = embed(name, g.texts)
    cos = np.einsum("ij,ij->i", emb[g.edges[:, 0]], emb[g.edges[:, 1]])
    np.save(path, cos)
    return cos


# --------------------------------------------------------------------- edge selection
def budget(m, fraction):
    return int(round(fraction * m))


def select_band(cos, band, k):
    """Edges inside the cosine band, closest to its midpoint first (ties: lower edge ID)."""
    lo, hi = band
    inside = np.flatnonzero((cos >= lo) & (cos <= hi))
    order = np.lexsort((inside, np.abs(cos[inside] - (lo + hi) / 2)))
    return inside[order][:k]


def select_random(m, k, seed=0):
    return np.sort(np.random.default_rng(seed).choice(m, size=k, replace=False))


def variants(name, cos):
    """Every weight variant in the study. A variant with "edges" needs LLM scores."""
    m = len(cos)
    llm = {"model": config.LLM["model"], "provider": config.LLM["provider"], "prompt": config.LLM["prompt"]}
    main = select_band(cos, config.BAND, budget(m, config.BUDGET))
    out = [{"variant": "W0", "suite": "main"},
           {"variant": "W1", "suite": "main"},
           {"variant": "W2", "suite": "main", "edges": main, **llm}]
    if name == "synthetic":
        return [{**v, "suite": "sanity"} for v in out]
    # Ablation: is the band selection better than random edges / other bands?
    out.append({"variant": "W2-random", "suite": "ablation", "edges": select_random(m, len(main)), **llm})
    for band in config.ABLATION_BANDS:
        out.append({"variant": f"W2-band{band[0]}-{band[1]}", "suite": "ablation",
                    "edges": select_band(cos, band, budget(m, config.BUDGET)), **llm})
    # Ablation on one dataset: other prompt wording, bigger model (same edges as W2)
    if name == config.ABLATION_DATASET:
        for p in config.ABLATION_PROMPTS:
            out.append({"variant": f"W2-prompt-{p}", "suite": "ablation", "edges": main, **llm, "prompt": p})
        for model, provider, tag in config.ABLATION_MODELS:
            out.append({"variant": f"W2-{tag}", "suite": "ablation", "edges": main,
                        **llm, "model": model, "provider": provider})
    # Sensitivity: how many LLM calls are worth paying for?
    for f in config.BUDGET_SWEEP:
        if f != config.BUDGET:
            out.append({"variant": f"W2-{round(f * 100)}pct", "suite": "sensitivity",
                        "edges": select_band(cos, config.BAND, budget(m, f)), **llm})
    return out


# --------------------------------------------------------------------- weights
def score_file(name, variant):
    return SCORES_DIR / f"{name}__{variant}.csv"


def weights_for(name, v, cos):
    if v["variant"] == "W0":
        return np.ones(len(cos))
    w = np.maximum(cos, config.WEIGHT_FLOOR)
    if v["variant"] == "W1":
        return w
    path = score_file(name, v["variant"])
    if not path.exists():
        raise FileNotFoundError(f"{path.name} missing: run `make llm` first")
    s = pd.read_csv(path)
    ok = s["score"].notna()  # a failed call keeps its W1 weight (never becomes 0)
    w[s.loc[ok, "edge_id"].to_numpy()] = np.maximum(s.loc[ok, "score"].to_numpy() / 10, config.WEIGHT_FLOOR)
    return w


def run():
    log = {}
    for name in config.DATASETS + ["synthetic"]:
        cos = cosine(name)
        lo, hi = config.BAND
        sec = config.PROCESSED_DIR / f"{name}_emb_{config.EMBED_REVISION[:8]}.sec"
        log[name] = {"encode_seconds": float(sec.read_text()) if sec.exists() else None, "edges": len(cos),
                     "cos_mean": round(float(cos.mean()), 4), "in_band": int(((cos >= lo) & (cos <= hi)).sum()),
                     "llm_budget": budget(len(cos), config.BUDGET)}
        print(f"   {name}: {log[name]}")
    (config.OUTPUT_DIR / "embedding_log.json").write_text(json.dumps(log, indent=2))
