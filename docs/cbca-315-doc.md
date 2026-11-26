# CBCA315 Minor Project: Requirements (merged)

Source: *Minor Project Research Pipeline: Community Detection in Social Networks*, Dr. Soumita Das, Bennett University (38 pages). The PDF header says CSET347; this project folder uses CBCA315. Only the parts that apply to Topic 2 are kept here.

## Milestones (Tuesdays and Thursdays only)

| Milestone | Date | Evidence |
|---|---|---|
| Overleaf, GitHub, Colab links shared | Tue 29 Sep 2026 | Links |
| CP1 Problem analysis + design approval | Tue 6 Oct | Literature matrix (≥15 papers, ≥10 from 2020-2026), gap, research question, objectives, base paper, datasets + metrics, pseudocode, baselines, protocol |
| **CP2 Coding + experimental results** | **Thu 15 Oct** | Baselines + method running from GitHub/Colab, synthetic sanity test, results tables (mean ± std, 5 seeds), ablation, sensitivity, runtime, downstream task, significance tests. Tag `v1.0-results-freeze` |
| CP3 Full draft review | Thu 5 Nov | Complete draft in the journal template |
| **CP4 Final paper to instructor** | **Tue 10 Nov** | Final PDF + Overleaf, GitHub, Colab links, similarity report, chosen journal + backup |
| Feedback / journal submission | Thu 12 Nov; submit by Sun 15 Nov | Instructor approval; journal acknowledgement with manuscript ID |
| Viva | Tue 17 / Thu 19 Nov | Presentation + live code walkthrough |

## Marks (30)

| Component | Marks | Assessed at |
|---|---|---|
| Problem Analysis (literature, gap, research question) | 6 | CP1 |
| Project Design (method, base paper, protocol) | 6 | CP1 |
| **Project Implementation (core)**: reproducible code, baselines + method + all experiments, seeds, ablation, sensitivity, significance | 6 | CP2, re-checked at viva |
| Project Report: structure, results, referencing, similarity | 6 | CP4 |
| **Communication (core)**: journal submission with manuscript ID (3), viva (2), checkpoint reporting (1) | 6 | 15-19 Nov |

Results that cannot be regenerated from the repository are not credited. Journal acceptance is not required.

## Topic 2: Information Diffusion Using LLMs for Community Detection (Section 9)

* **Idea:** give every edge a weight from the two papers' text; sentence embeddings for all edges, a small LLM only for "ambiguous" edges (cosine in [0.3, 0.6]); run diffusion-style detectors (label propagation, Leiden, Infomap).
* **Variants:** W0 structure only, W1 SBERT cosine, W2 = W1 + LLM re-scoring.
* **Datasets:** Cora, PubMed with raw text (TAPE release). Synthetic graph with planted communities as a sanity check.
* **Metrics:** NMI, ARI, ACC (Hungarian), macro-F1, modularity; cost = LLM calls, tokens, time, model name/size.
* **Report must contain:** dataset stats with text length; NMI/ARI/ACC/F1 for W0, W1, W2 × 3 detectors (mean ± std); accuracy vs cost; effect of the band; prompt and model ablation; case study of re-weighted edges.
* **Pitfalls:** label propagation is random, so average over seeds; report exact prompt and model; temperature 0; never send whole abstracts for every edge.

## Common rules (Sections 5, 6, 12, 13)

* Seeds 0-4 for every method; seed logged in every CSV row; every number traceable to the results folder.
* Shared evaluation for all methods (Section 6.1); modularity always reported.
* Downstream task (Section 6.2): Independent Cascade, k ∈ {5, 10, 20, 50}, p = 0.05, 200 runs, community seeds vs top-degree seeds, on two datasets.
* Labels are for evaluation only, never for tuning.
* Paper: title ≤ 15 words, abstract 150-250 words, Introduction, Related Work (with comparison table), Preliminaries, Method (pseudocode, complexity), Experimental Setup, Results, Limitations + Conclusion, Declarations (incl. AI-use disclosure).
* Generative AI only where the journal permits, disclosed. Verify every reference. Similarity under 10-15%.

## Target journals (Section 14)

IEEE Transactions on Computational Social Systems (IEEEtran template); ACM TKDD (acmart); Expert Systems with Applications (elsarticle). Submit to one journal at a time, through the official portal only.

## Pre-submission checklist (Appendix C)

- [ ] Journal template, length limits, reference style
- [ ] Every table/figure referenced; self-contained captions
- [ ] All numbers match the files in `outputs/`
- [ ] Datasets and metrics match the base paper; deviations stated
- [ ] Every reference verified via DOI / Google Scholar
- [ ] Similarity report checked and shared
- [ ] Declarations: funding, conflicts, data and code availability, author contributions, AI use
- [ ] GitHub public or archived; Colab notebook runs from a clean session
- [ ] Cover letter; ORCID ready
- [ ] Final paper to instructor at CP4; approval received
