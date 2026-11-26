"""Step 5 (downstream task): do better communities spread information further?

Independent Cascade (IC) with probability p, k seed nodes (course guide, Section 6.2).
Strategies: top-k degree, and community seeds from W0 / W1 / W2 communities.

IC on an undirected graph equals the "live-edge" model: each edge is live with
probability p, and every node connected to a seed through live edges becomes
active. We sample the live edges ONCE per simulation and reuse them for every
strategy, so all strategies face identical diffusion conditions (fair comparison).
"""
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

import config
import detect
import edge_weights
import load_data


def live_edge_samples(n, edges, p, sims, seed):
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(sims):
        live = edges[rng.random(len(edges)) < p]
        adj = coo_matrix((np.ones(len(live)), (live[:, 0], live[:, 1])), shape=(n, n))
        _, comp = connected_components(adj, directed=False)
        samples.append((comp, np.bincount(comp)))
    return samples


def spread(seeds, samples):
    """Number of activated nodes in each simulation."""
    return np.array([sizes[np.unique(comp[seeds])].sum() for comp, sizes in samples])


def degree_seeds(degree, k):
    return np.lexsort((np.arange(len(degree)), -degree))[:k]


def community_seeds(membership, degree, k):
    """k seeds spread over communities by size, highest degree first (guide code).

    Always exactly k distinct nodes: unused slots go to the top-degree nodes left.
    """
    n = len(membership)
    comms = sorted((np.flatnonzero(membership == c) for c in np.unique(membership)), key=lambda c: (-len(c), c[0]))
    seeds = []
    for c in comms:
        quota = max(1, round(k * len(c) / n))
        ranked = c[np.lexsort((c, -degree[c]))]
        seeds += [int(v) for v in ranked if v not in seeds][:quota]
        if len(seeds) >= k:
            break
    for v in degree_seeds(degree, n):
        if len(seeds) >= k:
            break
        if int(v) not in seeds:
            seeds.append(int(v))
    return np.array(seeds[:k])


def run():
    d = config.DIFFUSION
    rows = []
    for name in config.DATASETS:
        g = load_data.load(name)
        cos = edge_weights.cosine(name)
        degree = np.bincount(g.edges.ravel(), minlength=g.n)
        samples = live_edge_samples(g.n, g.edges, d["p"], d["simulations"], d["sim_seed"])
        memberships = {v["variant"]: detect.detect(d["detector"], g.n, g.edges,
                                                    edge_weights.weights_for(name, v, cos), d["community_seed"])
                       for v in edge_weights.variants(name, cos) if v["suite"] == "main"}
        for k in d["k"]:
            strategies = {"degree": degree_seeds(degree, k),
                          **{f"community-{var}": community_seeds(mem, degree, k) for var, mem in memberships.items()}}
            per_sim = {}
            for strategy, seeds in strategies.items():
                assert len(set(seeds.tolist())) == k
                per_sim[strategy] = s = spread(seeds, samples)
                rows.append({"dataset": name, "k": k, "strategy": strategy, "spread_mean": s.mean(),
                             "spread_std": s.std(ddof=1), "spread_pct": 100 * s.mean() / g.n})
            # Paired test over the same 200 simulations: W2 communities vs W1 communities
            a, b = per_sim["community-W2"], per_sim["community-W1"]
            rows[-1]["p_W2_vs_W1"] = wilcoxon(a, b).pvalue if np.any(a != b) else 1.0
        print(f"   {name}: done")
    df = pd.DataFrame(rows)
    df.to_csv(config.OUTPUT_DIR / "diffusion.csv", index=False)
    print(df.pivot_table(index=["dataset", "k"], columns="strategy", values="spread_mean").round(1).to_string())
