# 02. Decisions Log

| # | Decision | Reason |
|---|---|---|
| D1 | Topic 2, Section 9 plan of the course guide: W0 / W1 / W2 edge weights × LPA / Leiden / Infomap | Your choice. Follows the guide's default execution plan |
| D2 | Cora + PubMed with original title/abstract text from the TAPE release (He et al., ICLR 2024) | The datasets the guide names for this topic. TAPE is the data reference, not a clustering baseline (it is supervised classification) |
| D3 | Text is joined to graph nodes **by paper ID**, never by row position | TAPE's PubMed loader pairs by row and shifts every row after index 2459 by one; its Cora loader can reuse the previous paper's abstract |
| D4 | PubMed paper PMID 17874530 excluded (19,716 nodes, 44,323 edges) | It has no title or abstract in the release. Same graph for every method |
| D5 | Graphs made undirected; self-loops and duplicate edges removed; all nodes kept (no largest-component filter) | Course guide data hygiene (Section 7.4); identical for every method |
| D6 | Embeddings: `all-MiniLM-L6-v2`, frozen, pinned revision `1110a243` | Small, CPU-friendly, no training. The guide's starter code uses it |
| D7 | LLM: `meta-llama/llama-3.1-8b-instruct` on OpenRouter, provider pinned to `deepinfra/fp8`, no fallback, temperature 0 | Your choice of OpenRouter; a 7-8B open model as the guide asks; a pinned endpoint keeps the method fixed |
| D8 | LLM budget = 10% of edges, chosen from the cosine band [0.3, 0.6], closest to the band midpoint first | Guide's band. The earlier 2,000-edge cap was dropped: it made the PubMed budget sweep identical at every level, and the $1 spending ceiling already limits cost |
| D9 | Only a whole number 0-10 counts as an answer; a failed call keeps its W1 weight; nothing becomes 0 by mistake | Honest cost and quality numbers |
| D10 | Every LLM attempt logged to `outputs/llm/calls.jsonl` and cached; hard stop at $1.00 | Reproducible, no double payment |
| D11 | Prompt = the guide's wording ("how likely is information shared by A to be adopted by B..."); ablation uses a plain "topic similarity" wording | Guide asks for a prompt ablation |
| D12 | Model ablation: Llama 3.1 70B on Cora, same edges | Guide asks to compare two model sizes "if possible"; cost < $0.10 |
| D13 | Ablations: random edges (same number of calls), shifted bands [0.2, 0.5] and [0.4, 0.7]. Sensitivity: budget 5 / 10 / 20% | Tests whether the band selection matters and how much budget pays off |
| D14 | Macro-F1 averages over the real classes only | The guide's starter code would count every extra small community as a 0-score class |
| D15 | Modularity always measured on the original unweighted graph | Comparable across W0 / W1 / W2 |
| D16 | Downstream: Independent Cascade via shared live-edge samples, exactly k distinct seeds, Leiden (seed 0) communities | Identical diffusion conditions for every strategy; fair comparison |
| D17 | Significance: Wilcoxon signed-rank, paired by seed (n = 5) and pooled over datasets × detectors (n = 30) | With n = 5 the smallest possible two-sided p is 0.0625, so the pooled test is declared up front |
| D18 | Python 3.13 + uv; one `Makefile`; flat `src/` modules run by `src/run_pipeline.py` | Your requirement: simple, same layout as cbca-293 |
| D19 | Reports in Markdown, one file per chapter, in `reports/` | Your requirement, same as cbca-293 |
| D20 | Full report draft written with AI help, disclosed in the declarations | Your instruction; the course guide requires disclosure (p. 34) |
| D21 | No automated tests, no dashboard, no database | Your instruction: keep it simple |
