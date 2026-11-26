"""Step 3: ask a small LLM (Llama 3.1 8B via OpenRouter) to score selected edges 0-10.

* Every attempt is appended to outputs/llm/calls.jsonl (tokens, cost, latency, answer).
* A valid or invalid answer is cached: re-running never pays twice.
* A network failure is NOT a score of zero. It is logged and retried next run.
* Spending stops at config.SPEND_CEILING_USD.
"""
import hashlib
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

import config
import edge_weights
import load_data

PROMPTS = {
    # Course guide wording (Section 9 starter code).
    "influence": ("Two connected users/documents:\nA: {a}\nB: {b}\n"
                  "On a scale of 0 to 10, how likely is information shared by A to be adopted by B "
                  "because they are about the same topic? Reply with one integer only."),
    # Ablation: plain topic-similarity wording.
    "topic": ("Two research papers:\nA: {a}\nB: {b}\n"
              "On a scale of 0 to 10, how closely related are the research topics of A and B? "
              "Reply with one integer only."),
}
LEDGER = config.LLM_DIR / "calls.jsonl"


class BudgetExceeded(RuntimeError):
    pass


def parse_score(text):
    """Accept only a whole number 0-10 (optionally followed by a full stop)."""
    m = re.fullmatch(r"\s*(\d{1,2})\.?\s*", text or "")
    return int(m.group(1)) if m and int(m.group(1)) <= 10 else None


def read_ledger():
    if not LEDGER.exists():
        return []
    return [json.loads(l) for l in LEDGER.read_text().splitlines() if l.strip()]


def cost_of(row):
    """Real cost if the API returned it; else a conservative $1 per million tokens."""
    if row.get("cost") is not None:
        return row["cost"]
    return ((row.get("prompt_tokens") or 0) + (row.get("completion_tokens") or 0)) * 1e-6


class Scorer:
    def __init__(self, model, provider, prompt):
        load_dotenv(config.ROOT / ".env")
        self.key = os.environ.get("OPENROUTER_API_KEY", "")
        self.model, self.provider, self.prompt = model, provider, prompt
        ledger = read_ledger()
        self.cache = {r["key"]: r for r in ledger if r["status"] in ("ok", "invalid")}
        self.spent = sum(cost_of(r) for r in ledger)
        self.lock = threading.Lock()

    def call_api(self, text):
        if not self.key:
            raise SystemExit("OPENROUTER_API_KEY is missing. Put it in the .env file.")
        body = {"model": self.model, "messages": [{"role": "user", "content": text}],
                "temperature": config.LLM["temperature"], "max_tokens": config.LLM["max_tokens"],
                "provider": {"order": [self.provider], "allow_fallbacks": False, "require_parameters": True},
                "usage": {"include": True}}
        r = requests.post(config.LLM["url"], json=body, timeout=config.LLM["timeout_s"],
                          headers={"Authorization": f"Bearer {self.key}", "X-Title": "CBCA315 Semantic Diffusion"})
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
        return r.json()

    def log(self, row):
        with self.lock:
            with open(LEDGER, "a") as f:
                f.write(json.dumps(row) + "\n")
            self.spent += cost_of(row)
            if row["status"] in ("ok", "invalid"):
                self.cache[row["key"]] = row

    def score(self, a, b, tag):
        n = config.LLM["max_chars"]
        text = PROMPTS[self.prompt].format(a=a[:n], b=b[:n])
        key = hashlib.sha256(f"{self.model}|{self.provider}|{self.prompt}|{text}".encode()).hexdigest()
        if key in self.cache:
            return self.cache[key]
        row = {}
        for attempt in range(config.LLM["retries"] + 1):
            if self.spent >= config.SPEND_CEILING_USD:
                raise BudgetExceeded(f"spent ${self.spent:.4f}, ceiling ${config.SPEND_CEILING_USD}")
            t0 = time.time()
            row = {"key": key, "tag": tag, "model": self.model, "provider": self.provider, "prompt": self.prompt,
                   "attempt": attempt, "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
            try:
                resp = self.call_api(text)
                answer = resp["choices"][0]["message"]["content"]
                usage = resp.get("usage") or {}
                score = parse_score(answer)
                row.update(status="ok" if score is not None else "invalid", answer=answer, score=score,
                           prompt_tokens=usage.get("prompt_tokens"), completion_tokens=usage.get("completion_tokens"),
                           cost=usage.get("cost"), generation_id=resp.get("id"), served_by=resp.get("provider"),
                           latency_s=round(time.time() - t0, 3))
                self.log(row)
                return row
            except (BudgetExceeded, SystemExit):
                raise
            except Exception as exc:
                row.update(status="failed", score=None, error=str(exc)[:300], latency_s=round(time.time() - t0, 3))
                self.log(row)
                time.sleep(2 ** attempt)
        return row


def run(pilot=False):
    config.LLM_DIR.mkdir(parents=True, exist_ok=True)
    edge_weights.SCORES_DIR.mkdir(parents=True, exist_ok=True)
    names = ["cora"] if pilot else ["synthetic"] + config.DATASETS
    for name in names:
        g = load_data.load(name)
        cos = edge_weights.cosine(name)
        for v in edge_weights.variants(name, cos):
            if "edges" not in v:
                continue
            edges = v["edges"][:config.PILOT_EDGES] if pilot else v["edges"]
            scorer = Scorer(v["model"], v["provider"], v["prompt"])
            tag = f"{name}__{v['variant']}"
            pairs = [(g.texts[g.edges[e, 0]], g.texts[g.edges[e, 1]]) for e in edges]
            try:
                with ThreadPoolExecutor(config.LLM["workers"]) as pool:
                    rows = list(pool.map(lambda p: scorer.score(p[0], p[1], tag), pairs))
            except BudgetExceeded as exc:
                raise SystemExit(f"STOPPED, spending ceiling reached: {exc}")
            df = pd.DataFrame({"rank": np.arange(len(edges)), "edge_id": edges, "cos": cos[edges].round(4),
                               "score": [r.get("score") for r in rows], "status": [r.get("status") for r in rows]})
            counts = df["status"].value_counts().to_dict()
            if pilot:
                print(f"   pilot on {len(df)} Cora edges: {counts}")
                print(f"   score distribution: {df['score'].value_counts().sort_index().to_dict()}")
                print(f"   Spearman(score, cosine) = {df[['score', 'cos']].corr('spearman').iloc[0, 1]:.3f}")
                break
            df.to_csv(edge_weights.score_file(name, v["variant"]), index=False)
            print(f"   {tag:32s} edges={len(df):5d} {counts}")
    ledger = read_ledger()
    print(f"   ledger: {len(ledger)} API attempts, total spent ${sum(cost_of(r) for r in ledger):.4f}")
