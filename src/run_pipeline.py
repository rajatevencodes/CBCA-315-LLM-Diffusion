"""Run the whole project, or one step:

    uv run python src/run_pipeline.py                 # every step, in order
    uv run python src/run_pipeline.py --step llm      # one step

Steps: data -> weights -> pilot -> llm -> experiments -> diffusion -> charts
"""
import argparse
import time

import config
import detect
import diffusion
import edge_weights
import llm_scoring
import load_data
import make_charts

STEPS = {
    "data": ("1. Download + align Cora and PubMed", load_data.run),
    "weights": ("2. MiniLM embeddings -> W0 / W1 edge weights", edge_weights.run),
    "pilot": ("3a. LLM pilot on 50 Cora edges", lambda: llm_scoring.run(pilot=True)),
    "llm": ("3b. LLM scores for every selected edge (W2)", llm_scoring.run),
    "experiments": ("4. Community detection, 3 detectors x 5 seeds", detect.run),
    "diffusion": ("5. Downstream influence spread (Independent Cascade)", diffusion.run),
    "charts": ("6. Tables, significance tests, charts", make_charts.run),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--step", choices=STEPS, help="run only this step")
    args = ap.parse_args()
    config.OUTPUT_DIR.mkdir(exist_ok=True)
    for key in [args.step] if args.step else [k for k in STEPS if k != "pilot"]:
        title, fn = STEPS[key]
        print(f"\n== {title}")
        t0 = time.time()
        fn()
        print(f"   ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
