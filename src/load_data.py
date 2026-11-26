"""Step 1: download, verify, align text to graph nodes by paper ID, and load.

A dataset is a `Graph`: node IDs, labels (used ONLY for evaluation), raw text,
and an undirected edge list `edges[i] = (u, v)` with u < v. Row i is the edge ID.
"""
import hashlib
import json
import urllib.request
import zipfile
from collections import defaultdict
from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd

import config


@dataclass
class Graph:
    name: str
    ids: list
    labels: np.ndarray
    texts: list
    edges: np.ndarray  # shape (m, 2)

    @property
    def n(self):
        return len(self.ids)

    @property
    def m(self):
        return len(self.edges)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download(name):
    """Download the archive once, then always check its checksum."""
    fname, drive_id, expected = config.SOURCES[name]
    path = config.RAW_DIR / fname
    if not path.exists():
        config.RAW_DIR.mkdir(parents=True, exist_ok=True)
        print(f"   downloading {fname} ...")
        url = f"https://drive.usercontent.google.com/download?id={drive_id}&export=download&confirm=t"
        urllib.request.urlretrieve(url, path)
    digest = sha256(path)
    if digest != expected:
        raise ValueError(f"{fname}: checksum mismatch ({digest})")
    return path, digest


def undirected(pairs, n):
    """Remove self-loops and duplicate edges; return sorted (u, v) with u < v."""
    e = {(min(a, b), max(a, b)) for a, b in pairs if a != b}
    return np.array(sorted(e), dtype=np.int64).reshape(-1, 2)


def parse_cora(zip_path):
    z = zipfile.ZipFile(zip_path)
    rows = [l.split("\t") for l in z.read("cora_orig/cora.content").decode().splitlines()]
    ids = [r[0] for r in rows]
    classes = sorted({r[-1] for r in rows})
    labels = np.array([classes.index(r[-1]) for r in rows])
    index = {pid: i for i, pid in enumerate(ids)}
    cites = [l.split() for l in z.read("cora_orig/cora.cites").decode().splitlines()]
    edges = undirected([(index[a], index[b]) for a, b in cites], len(ids))

    # Text is joined by paper ID. Some IDs list several extraction files: take the
    # one with the most complete Title/Abstract. Never borrow another paper's text.
    files = defaultdict(list)
    for line in z.read("cora_orig/mccallum/cora/papers").decode("latin-1").splitlines():
        pid, fn = line.split("\t")[:2]
        if fn not in files[pid]:
            files[pid].append(fn)
    names, pre = set(z.namelist()), "cora_orig/mccallum/cora/extractions/"

    def title_abstract(fn):
        if pre + fn not in names:
            return "", ""
        lines = z.read(pre + fn).decode("latin-1").splitlines()
        ti = next((l[6:].strip() for l in lines if l.startswith("Title:")), "")
        ab = next((l[9:].strip() for l in lines if l.startswith("Abstract:")), "")
        return ti, ab

    texts, coverage = [], defaultdict(int)
    for pid in ids:
        ti, ab = max((title_abstract(f) for f in files[pid]), key=lambda t: (bool(t[0]) + bool(t[1]), len(t[1])))
        coverage[f"title={bool(ti)}, abstract={bool(ab)}"] += 1
        texts.append(f"Title: {ti}\nAbstract: {ab}")
    return ids, labels, texts, edges, classes, {"text_coverage": dict(coverage)}


def parse_pubmed(zip_path):
    z = zipfile.ZipFile(zip_path)
    node_lines = z.read("PubMed_orig/data/Pubmed-Diabetes.NODE.paper.tab").decode().splitlines()[2:]
    all_ids = [l.split("\t")[0] for l in node_lines]
    all_labels = [int(l.split("\t")[1].split("=")[1]) - 1 for l in node_lines]
    # Text is joined by PMID. (TAPE pairs text by row position, which shifts every
    # row after the one record that has no PMID.)
    text_by_id = {str(r["PMID"]): f"Title: {r.get('TI') or ''}\nAbstract: {r.get('AB') or ''}"
                  for r in json.loads(z.read("PubMed_orig/pubmed.json"))
                  if r.get("PMID") and (r.get("TI") or r.get("AB"))}
    keep = [pid for pid in all_ids if pid in text_by_id]
    index = {pid: i for i, pid in enumerate(keep)}
    labels = np.array([lab for pid, lab in zip(all_ids, all_labels) if pid in index])
    pairs, dropped = [], 0
    for l in z.read("PubMed_orig/data/Pubmed-Diabetes.DIRECTED.cites.tab").decode().splitlines()[2:]:
        p = l.split("\t")
        a, b = p[1].split(":")[1], p[3].split(":")[1]
        if a in index and b in index:
            pairs.append((index[a], index[b]))
        else:
            dropped += 1
    classes = ["Diabetes Mellitus, Experimental", "Diabetes Mellitus Type 1", "Diabetes Mellitus Type 2"]
    extra = {"text_coverage": {"with_text": len(keep), "no_text_excluded": len(all_ids) - len(keep)},
             "excluded_ids": [p for p in all_ids if p not in index], "edges_dropped_with_excluded_node": dropped}
    return keep, labels, [text_by_id[p] for p in keep], undirected(pairs, len(keep)), classes, extra


def prepare(name):
    """Download + verify + align; write processed CSVs; return a manifest entry."""
    path, digest = download(name)
    ids, labels, texts, edges, classes, extra = (parse_cora if name == "cora" else parse_pubmed)(path)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": ids, "label": labels, "text": texts}).to_csv(config.PROCESSED_DIR / f"{name}_nodes.csv", index=False)
    pd.DataFrame(edges, columns=["u", "v"]).to_csv(config.PROCESSED_DIR / f"{name}_edges.csv", index=False)
    return {"archive": path.name, "sha256": digest, "nodes": len(ids), "edges": len(edges), "classes": classes, **extra}


TOPICS = [["cricket", "match", "score"], ["election", "vote", "policy"],
          ["movie", "actor", "trailer"], ["exam", "result", "college"]]


def synthetic():
    """Planted-partition graph with topic words per community (course guide, Section 7.3)."""
    s = config.SYNTHETIC
    p = np.full((s["blocks"], s["blocks"]), s["p_out"])
    np.fill_diagonal(p, s["p_in"])
    g = nx.stochastic_block_model([s["block_size"]] * s["blocks"], p.tolist(), seed=s["seed"])
    rng = np.random.default_rng(s["seed"])
    labels = np.array([g.nodes[v]["block"] for v in g])
    texts = [" ".join(rng.choice(TOPICS[b], size=s["words_per_node"])) for b in labels]
    return Graph("synthetic", [str(v) for v in g], labels, texts, undirected(g.edges(), g.number_of_nodes()))


def load(name):
    if name == "synthetic":
        return synthetic()
    nodes = pd.read_csv(config.PROCESSED_DIR / f"{name}_nodes.csv", dtype={"id": str}, keep_default_na=False)
    edges = pd.read_csv(config.PROCESSED_DIR / f"{name}_edges.csv").to_numpy(dtype=np.int64)
    return Graph(name, nodes["id"].tolist(), nodes["label"].to_numpy(), nodes["text"].tolist(), edges)


def run():
    manifest, stats = {}, []
    for name in config.DATASETS:
        manifest[name] = prepare(name)
    for name in config.DATASETS + ["synthetic"]:
        g = load(name)
        words = [len(t.split()) for t in g.texts]
        stats.append({"dataset": name, "nodes": g.n, "edges": g.m, "classes": len(set(g.labels)),
                      "avg_degree": round(2 * g.m / g.n, 2), "avg_words": round(float(np.mean(words)), 1)})
    config.OUTPUT_DIR.mkdir(exist_ok=True)
    (config.OUTPUT_DIR / "data_manifest.json").write_text(json.dumps(manifest, indent=2))
    pd.DataFrame(stats).to_csv(config.OUTPUT_DIR / "dataset_stats.csv", index=False)
    print(pd.DataFrame(stats).to_string(index=False))
