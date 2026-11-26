"""Step 4: run the community detectors on every weight variant and score them.

Detectors (all weighted, all diffusion/flow based):
  lpa      asynchronous label propagation (NetworkX)
  leiden   Leiden, modularity objective  (igraph)
  infomap  Infomap, 10 trials            (igraph)

Metrics (course guide, Section 6.1): NMI, ARI, Hungarian ACC, macro-F1, plus
modularity Q measured on the ORIGINAL unweighted graph for every method.
Results go to outputs/runs.csv, one row per run. Finished runs are skipped,
so an interrupted run continues where it stopped.
"""
import random
import resource
import sys
import time

import igraph as ig
import networkx as nx
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, f1_score, normalized_mutual_info_score

import config
import edge_weights
import load_data

RUNS_CSV = config.OUTPUT_DIR / "runs.csv"


# --------------------------------------------------------------------- detectors
def detect(name, n, edges, weights, seed):
    """Return one community id per node."""
    random.seed(seed)  # igraph draws its random numbers from Python's `random`
    np.random.seed(seed)
    if name == "lpa":
        g = nx.Graph()
        g.add_nodes_from(range(n))
        g.add_weighted_edges_from((int(u), int(v), float(w)) for (u, v), w in zip(edges, weights))
        membership = np.empty(n, dtype=np.int64)
        comms = sorted((sorted(c) for c in nx.community.asyn_lpa_communities(g, weight="weight", seed=seed)))
        for cid, c in enumerate(comms):
            membership[c] = cid
        return membership
    g = ig.Graph(n=n, edges=edges.tolist())
    g.es["weight"] = list(weights)
    if name == "leiden":
        part = g.community_leiden(objective_function="modularity", weights="weight", n_iterations=-1)
    else:
        part = g.community_infomap(edge_weights="weight", trials=10)
    return np.array(part.membership)


# --------------------------------------------------------------------- metrics
def cluster_acc(y_true, y_pred):
    """Best one-to-one matching of communities to classes (Hungarian) -> ACC, macro-F1.

    Communities left unmatched count as wrong. Macro-F1 averages over the real classes.
    """
    k = max(y_true.max(), y_pred.max()) + 1
    M = np.zeros((k, k), dtype=np.int64)
    np.add.at(M, (y_pred, y_true), 1)
    rows, cols = linear_sum_assignment(-M)
    y_mapped = dict(zip(rows, cols))
    y_mapped = np.array([y_mapped[p] for p in y_pred])
    return (float((y_mapped == y_true).mean()),
            float(f1_score(y_true, y_mapped, labels=np.unique(y_true), average="macro")))


def evaluate(y_true, membership, n, edges):
    acc, f1 = cluster_acc(y_true, membership)
    return {"NMI": normalized_mutual_info_score(y_true, membership, average_method="arithmetic"),
            "ARI": adjusted_rand_score(y_true, membership), "ACC": acc, "F1": f1,
            "Q": ig.Graph(n=n, edges=edges.tolist()).modularity(list(membership)),
            "n_comms": len(np.unique(membership))}


def peak_memory_mb():
    """Peak memory of this process so far (macOS reports bytes, Linux kilobytes)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(peak / 2**20 if sys.platform == "darwin" else peak / 2**10, 1)


# --------------------------------------------------------------------- experiments
def run(suites=("sanity", "main", "ablation", "sensitivity"), time_limit=None):
    """Run every (dataset, variant, detector, seed) in the chosen suites."""
    start = time.time()
    done = set()
    if RUNS_CSV.exists():
        old = pd.read_csv(RUNS_CSV)
        done = set(zip(old.dataset, old.variant, old.detector, old.seed))
    for name in ["synthetic"] + config.DATASETS:
        g = load_data.load(name)
        cos = edge_weights.cosine(name)
        for v in edge_weights.variants(name, cos):
            if v["suite"] not in suites:
                continue
            try:
                w = edge_weights.weights_for(name, v, cos)
            except FileNotFoundError as exc:
                print(f"   skip {name} {v['variant']}: {exc}")
                continue
            for det in config.DETECTORS:
                for seed in config.SEEDS:
                    if (name, v["variant"], det, seed) in done:
                        continue
                    if time_limit and time.time() - start > time_limit:
                        print("   time limit reached; run again to continue")
                        return False
                    t0 = time.time()
                    membership = detect(det, g.n, g.edges, w, seed)
                    secs = time.time() - t0
                    row = {"dataset": name, "suite": v["suite"], "variant": v["variant"], "detector": det,
                           "seed": seed, **evaluate(g.labels, membership, g.n, g.edges),
                           "time_s": round(secs, 3), "peak_mem_mb": peak_memory_mb(),
                           "llm_edges": len(v.get("edges", []))}
                    pd.DataFrame([row]).to_csv(RUNS_CSV, mode="a", header=not RUNS_CSV.exists(), index=False)
                    print(f"   {name:9s} {v['variant']:16s} {det:8s} seed={seed} "
                          f"NMI={row['NMI']:.3f} comms={row['n_comms']} {secs:.1f}s")
    return True
