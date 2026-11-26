"""Step 6: turn the raw results into report tables, significance tests and charts.

Reads   outputs/runs.csv, outputs/diffusion.csv, outputs/llm/
Writes  outputs/tables/*.md (paste-ready tables), outputs/significance.csv,
        outputs/charts/*.png, outputs/summary.json
Nothing in the report is typed by hand: every number comes from these files.
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import config
import edge_weights
import load_data
from llm_scoring import cost_of, read_ledger

DET_NAMES = {"lpa": "LPA", "leiden": "Leiden", "infomap": "Infomap"}
COLORS = {"W0": "#9aa5b1", "W1": "#4c78a8", "W2": "#e45756", "W2-random": "#f2a541"}


def pm(x):
    """'mean ± std' over the 5 seeds."""
    return f"{x.mean():.3f} ± {x.std(ddof=1):.3f}"


def md_table(df, path, floatfmt=".3f"):
    path.write_text(df.to_markdown(index=False, floatfmt=floatfmt) + "\n")


def rank_biserial(diff):
    """Effect size for the signed-rank test: +1 = W2 always better, -1 = always worse."""
    diff = diff[diff != 0]
    if len(diff) == 0:
        return 0.0
    ranks = pd.Series(np.abs(diff)).rank().to_numpy()
    return float((ranks[diff > 0].sum() - ranks[diff < 0].sum()) / ranks.sum())


def paired_test(runs, a, b, metric="NMI"):
    """Wilcoxon signed-rank of variant a vs b, paired by (dataset, detector, seed)."""
    key = ["dataset", "detector", "seed"]
    x = runs[runs.variant == a].set_index(key)[metric]
    y = runs[runs.variant == b].set_index(key)[metric]
    x, y = x.align(y, join="inner")
    d = (x - y).to_numpy()
    p = wilcoxon(x, y).pvalue if np.any(d != 0) else 1.0
    return {"n": len(d), "mean_diff": d.mean(), "wins": int((d > 0).sum()), "losses": int((d < 0).sum()),
            "p_value": p, "rank_biserial": rank_biserial(d)}


def llm_costs():
    """Measured cost per call (per model) and the cost each variant would have alone."""
    led = pd.DataFrame(read_ledger())
    led["usd"] = [cost_of(r) for r in read_ledger()]
    ok = led[led.status.isin(["ok", "invalid"])]
    per_model = ok.groupby("model").agg(calls=("usd", "size"), usd_per_call=("usd", "mean"),
                                        tokens_in=("prompt_tokens", "mean"), latency_s=("latency_s", "mean"))
    totals = {"api_attempts": len(led), "failed_attempts": int((led.status == "failed").sum()),
              "invalid_answers": int((led.status == "invalid").sum()), "total_usd": float(led.usd.sum()),
              "total_tokens": int(led.prompt_tokens.fillna(0).sum() + led.completion_tokens.fillna(0).sum())}
    return per_model, totals


def run():
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    config.CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    runs = pd.read_csv(config.OUTPUT_DIR / "runs.csv")
    summary = {}

    # ---- Table 1: datasets
    stats = pd.read_csv(config.OUTPUT_DIR / "dataset_stats.csv")
    md_table(stats, config.TABLES_DIR / "t1_datasets.md", ".1f")

    # ---- Table 2: main results (mean ± std over 5 seeds)
    main = runs[runs.variant.isin(["W0", "W1", "W2"])]
    t2 = (main.groupby(["dataset", "detector", "variant"], sort=False)
          .agg(NMI=("NMI", pm), ARI=("ARI", pm), ACC=("ACC", pm), F1=("F1", pm), Q=("Q", pm),
               communities=("n_comms", "mean")).reset_index())
    t2["detector"] = t2.detector.map(DET_NAMES)
    md_table(t2, config.TABLES_DIR / "t2_main_results.md", ".0f")
    summary["main_nmi"] = main.groupby(["dataset", "detector", "variant"]).NMI.mean().round(4).unstack().to_dict("index")
    summary["main_nmi"] = {f"{d}/{k}": v for (d, k), v in summary["main_nmi"].items()}

    # ---- Significance tests (NMI), predeclared comparisons
    sig = []
    for a, b in [("W2", "W1"), ("W2", "W0"), ("W1", "W0"), ("W2", "W2-random")]:
        for (ds, det), grp in runs[runs.dataset != "synthetic"].groupby(["dataset", "detector"]):
            sig.append({"comparison": f"{a} vs {b}", "dataset": ds, "detector": DET_NAMES[det],
                        **paired_test(grp, a, b)})
        sig.append({"comparison": f"{a} vs {b}", "dataset": "all", "detector": "all",
                    **paired_test(runs[runs.dataset != "synthetic"], a, b)})
    sig = pd.DataFrame(sig)
    sig.to_csv(config.OUTPUT_DIR / "significance.csv", index=False)
    md_table(sig, config.TABLES_DIR / "t3_significance.md", ".4f")
    summary["pooled_tests"] = sig[sig.dataset == "all"].set_index("comparison").round(4).to_dict("index")

    # ---- Table 4: ablation (mean NMI per detector)
    abl = runs[(runs.suite.isin(["main", "ablation"])) & (runs.variant != "W0") & (runs.dataset != "synthetic")]
    t4 = abl.pivot_table(index=["dataset", "variant"], columns="detector", values="NMI", aggfunc="mean")
    t4 = t4.rename(columns=DET_NAMES).reset_index()
    md_table(t4, config.TABLES_DIR / "t4_ablation_nmi.md", ".4f")

    # ---- Table 5: budget sensitivity + cost (the accuracy vs cost view)
    per_model, totals = llm_costs()
    summary["llm_totals"] = totals
    usd_8b = per_model.loc[config.LLM["model"], "usd_per_call"]
    budgets = {"W1": 0.0, "W2-5pct": 0.05, "W2": 0.10, "W2-20pct": 0.20}
    sens = runs[runs.variant.isin(budgets) & (runs.dataset != "synthetic")].copy()
    sens["budget_pct"] = sens.variant.map(budgets) * 100
    t5 = sens.pivot_table(index=["dataset", "budget_pct", "llm_edges"], columns="detector", values="NMI",
                          aggfunc="mean").rename(columns=DET_NAMES).reset_index()
    t5["cost_usd"] = t5.llm_edges * usd_8b
    md_table(t5, config.TABLES_DIR / "t5_budget_vs_nmi.md", ".4f")

    # ---- Table 6: LLM cost per model + runtime
    pm_tab = per_model.reset_index()
    md_table(pm_tab, config.TABLES_DIR / "t6_llm_cost.md", ".6f")
    rt = runs[runs.variant.isin(["W0", "W1", "W2"])].groupby(["dataset", "detector"]).agg(
        time_s=("time_s", "mean"), peak_mem_mb=("peak_mem_mb", "max")).reset_index()
    emb = json.loads((config.OUTPUT_DIR / "embedding_log.json").read_text())
    rt["embed_s_once"] = rt.dataset.map(lambda d: emb[d]["encode_seconds"])
    md_table(rt, config.TABLES_DIR / "t7_runtime.md", ".2f")

    # ---- Table 8: downstream influence spread
    dif = pd.read_csv(config.OUTPUT_DIR / "diffusion.csv")
    t8 = dif.pivot_table(index=["dataset", "k"], columns="strategy", values="spread_mean").reset_index()
    t8 = t8.merge(dif.dropna(subset=["p_W2_vs_W1"])[["dataset", "k", "p_W2_vs_W1"]], on=["dataset", "k"])
    md_table(t8, config.TABLES_DIR / "t8_diffusion.md", ".2f")

    # ---- Sanity check on the synthetic graph
    syn = runs[runs.dataset == "synthetic"].pivot_table(index="detector", columns="variant", values="NMI",
                                                         aggfunc="mean").rename(index=DET_NAMES).reset_index()
    md_table(syn, config.TABLES_DIR / "t0_sanity_synthetic.md", ".3f")
    summary["sanity_nmi"] = syn.set_index("detector").round(4).to_dict("index")

    # ---- Case study: Cora edges whose weight the LLM changed the most
    g = load_data.load("cora")
    s = pd.read_csv(edge_weights.score_file("cora", "W2")).dropna(subset=["score"])
    s["change"] = s.score / 10 - s.cos
    s["same_class"] = g.labels[g.edges[s.edge_id, 0]] == g.labels[g.edges[s.edge_id, 1]]
    title = lambda i: g.texts[i].split("\n")[0].replace("Title: ", "")[:70]
    s["paper_A"] = [title(g.edges[e, 0]) for e in s.edge_id]
    s["paper_B"] = [title(g.edges[e, 1]) for e in s.edge_id]
    s["helpful"] = np.where(s.change > 0, s.same_class, ~s.same_class)
    case = pd.concat([s.nlargest(4, "change"), s.nsmallest(4, "change")])
    md_table(case[["edge_id", "cos", "score", "change", "same_class", "helpful", "paper_A", "paper_B"]],
             config.TABLES_DIR / "t9_case_study.md", ".2f")
    summary["llm_vs_class"] = {"up_weighted_same_class": float(s[s.change > 0].same_class.mean()),
                               "down_weighted_same_class": float(s[s.change < 0].same_class.mean()),
                               "all_band_edges_same_class": float(s.same_class.mean())}

    # ---- Charts
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=False)
    for ax, ds in zip(axes, config.DATASETS):
        d = main[main.dataset == ds]
        x = np.arange(len(config.DETECTORS))
        for i, var in enumerate(["W0", "W1", "W2"]):
            m = d[d.variant == var].groupby("detector").NMI
            ax.bar(x + (i - 1) * 0.26, [m.mean()[k] for k in config.DETECTORS], 0.26,
                   yerr=[m.std()[k] for k in config.DETECTORS], capsize=3, label=var, color=COLORS[var])
        ax.set_xticks(x, [DET_NAMES[k] for k in config.DETECTORS])
        ax.set_title(ds.capitalize())
        ax.set_ylabel("NMI (mean ± std, 5 seeds)")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(config.CHARTS_DIR / "fig1_main_nmi.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, ds in zip(axes, config.DATASETS):
        d = t5[t5.dataset == ds]
        for det in config.DETECTORS:
            ax.plot(d.llm_edges, d[DET_NAMES[det]], marker="o", label=DET_NAMES[det])
        ax.set_xlabel("LLM calls (edges re-scored)")
        ax.set_ylabel("mean NMI")
        ax.set_title(f"{ds.capitalize()}: quality vs LLM budget")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(config.CHARTS_DIR / "fig2_budget_vs_nmi.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, ds in zip(axes, config.DATASETS):
        cos = edge_weights.cosine(ds)
        ax.hist(cos, bins=60, color=COLORS["W1"])
        ax.axvspan(*config.BAND, color=COLORS["W2"], alpha=0.2, label="LLM band [0.3, 0.6]")
        ax.set_xlabel("cosine similarity of the two papers")
        ax.set_ylabel("edges")
        ax.set_title(ds.capitalize())
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(config.CHARTS_DIR / "fig3_cosine_band.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, ds in zip(axes, config.DATASETS):
        sc = pd.read_csv(edge_weights.score_file(ds, "W2")).dropna(subset=["score"])
        gg = load_data.load(ds)
        same = gg.labels[gg.edges[sc.edge_id, 0]] == gg.labels[gg.edges[sc.edge_id, 1]]
        rate = pd.Series(same).groupby(sc.score.to_numpy()).agg(["mean", "size"])
        ax.bar(rate.index, rate["mean"], color=COLORS["W2"])
        for xi, (r, nn) in rate.iterrows():
            ax.text(xi, r + 0.02, str(int(nn)), ha="center", fontsize=7)
        ax.set_xlabel("LLM score (0-10)")
        ax.set_ylabel("share of edges inside one class")
        ax.set_ylim(0, 1.1)
        ax.set_title(f"{ds.capitalize()}: does a higher LLM score mean same topic?")
    fig.tight_layout()
    fig.savefig(config.CHARTS_DIR / "fig4_llm_score_vs_class.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, ds in zip(axes, config.DATASETS):
        d = dif[dif.dataset == ds]
        for st, grp in d.groupby("strategy"):
            ax.plot(grp.k, grp.spread_mean, marker="o", label=st)
        ax.set_xlabel("k seed nodes")
        ax.set_ylabel("expected activated nodes")
        ax.set_title(f"{ds.capitalize()}: Independent Cascade, p = {config.DIFFUSION['p']}")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(config.CHARTS_DIR / "fig5_diffusion.png", dpi=200)
    plt.close(fig)

    (config.OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"   tables -> {config.TABLES_DIR.relative_to(config.ROOT)}, charts -> {config.CHARTS_DIR.relative_to(config.ROOT)}")
