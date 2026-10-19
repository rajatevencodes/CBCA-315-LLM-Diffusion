# Topic 2: project plan

Prepared 8 October 2026. Working project folder: `/Users/rajatsharma/Desktop/CBCA315-Semantic-Diffusion`.

## Selected project

**Budgeted LLM Refinement for Community Detection in Citation Networks**

Each node is a research paper. Each connection is a citation. Each paper has text, such as its title and abstract. We will find groups of related papers and test whether a few LLM judgments improve those groups.

The proposed question is: **Does refining a limited number of text-similarity weights with a small LLM improve community detection enough to justify its API cost?**

This adapts the instructor's Section 9 example. The contribution to investigate is the quality/cost trade-off under an explicit query budget, including a comparison with spending the same budget on randomly selected edges. The general idea of LLM edge weighting is already in the guide. Do not claim it is an original invention or promise an improvement before experiments.

## Confirmed choices and unresolved course rules

The student has chosen Topic 2, intends to do all work alone, requests uv and OpenRouter, and authorizes this Desktop folder. These choices are confirmed. The instructor has not yet been confirmed to have approved solo submission.

The supplied guide calls the course CSET347, the PDF filename refers to CBCA-215, and the student calls it CBCA315. Its applicability and current dates remain unconfirmed. The guide has group roles and a journal submission requirement. Doing the work alone does not automatically remove those obligations.

The requested technical choices are made below. Two academic decisions need instructor input:

1. Whether the solo arrangement, current deadline, and journal requirement apply as written.
2. Whether the two-dataset Section 9 adaptation and its comparison protocol are accepted in place of exact reproduction of a base paper's full setup.

The guide requires a base paper from 2022 to 2026 with matching datasets, metrics, and the base method as a baseline. The research did not identify a fully verified, simple two-dataset base paper that exactly matches this proposed method. TAPE is a supervised classification paper. ComGPT detects a local community around a seed. Neither paper's published scores can be used as direct evidence against our full-graph unsupervised clustering method.

Recommendation for the instructor: approve the Section 9 example on Cora and PubMed, use TAPE for original text/data provenance, use ComGPT as related work, and compare freshly run methods under one unsupervised protocol. Ask the instructor to confirm the accepted formal base paper and comparison scope. The complete evidence and a concrete approval request are in [base-paper-research.md](base-paper-research.md).

If this exception is refused, revise the base-paper choice and scope before building the full method. CDBNE is a closer unsupervised fallback candidate, but adds a third dataset and neural training, and its released evaluation code needs an audit. It is not silently included in the simple plan.

## Smallest useful implementation

| Decision | Selection |
| --- | --- |
| Real datasets | Cora and PubMed, original text from the TAPE release; pending dataset-scope approval and ID-alignment verification |
| Sanity dataset | One small stochastic block model with planted communities and simple generated topic text |
| Python | Python 3.12 and uv |
| Graphs | NetworkX and igraph |
| Detectors | Asynchronous label propagation, Leiden, Infomap |
| Text encoder | Frozen `sentence-transformers/all-MiniLM-L6-v2` |
| LLM | `meta-llama/llama-3.1-8b-instruct` through OpenRouter |
| API integration | A small `requests` client with validation, caching, and cost logging |
| Experiment records | YAML settings, CSV results, JSONL LLM cache |
| Analysis | NumPy, pandas, SciPy, scikit-learn, matplotlib |
| User interface | Three notebooks, plus scripts that regenerate the tables |
| Paper | Overleaf with the approved journal's current template |
| Proposed journal | IEEE Transactions on Computational Social Systems; ESWA as backup after checking its instructions |

No model training is planned. The sentence encoder runs locally or in Colab; the Llama model runs through the API. Local runtime still needs a benchmark before claiming a completion time. No web app, chatbot, RAG, vector database, login system, social-media scraping, or deployment is required for this plan.

## What the experiment compares

| Variant | Edge weights | Purpose |
| --- | --- | --- |
| W0 | Every existing edge has weight 1 | Structure-only baseline |
| W1 | Cosine similarity between original paper-text embeddings | Text baseline without LLM calls |
| W2 | W1 with selected uncertain-edge weights replaced by a validated LLM score / 10 | Proposed selective method |

All three variants use the same graph nodes and edges. Use a small fixed positive floor for weights so zero-similarity edges do not cause undefined or inconsistent detector behaviour. Record this preprocessing choice for every method.

The initial selection policy uses the guide's cosine band [0.3, 0.6]. It prioritizes edges closest to the band midpoint, with stable edge IDs breaking ties. Query at most the smaller of 10% of all edges, 2,000 edges, and the number of eligible edges. These are chosen starting settings, not experimentally validated optima. Freeze them before viewing final evaluation labels.

First run a pilot of at most 50 unique edges. Give the LLM only the two short text excerpts, with no class label or answer hint. Request one integer from 0 to 10. The score represents semantic affinity, not a calibrated probability of human information adoption.

Cache a response once and reuse it across all detectors and seeds. A failed request is a recorded failure; it must not turn into an invented score of zero. The full API implementation will add bounded retries, actual-usage logging, and an experiment spend ceiling. The proposal configuration currently does not implement these safeguards.

## Experiments we will keep

The proposed main table has 2 datasets x 3 weight variants x 3 detectors x 5 seeds = 90 detector runs. LLM responses are shared, so this does not imply 90 batches of API requests. This is the example-method comparison, not yet a substitute for any additional base-paper baseline required by the instructor.

- Main metrics: arithmetic NMI, ARI, one-to-one Hungarian-matched accuracy and macro-F1. Record community counts as well.
- Structural quality: modularity measured on the same original unweighted graph for every method.
- Resources: runtime, peak process memory, API attempts, tokens, and actual API cost. Separate first-run embedding/API time from cached detector time.
- Seeds: 0 through 4, with mean and standard deviation. Save node assignments and raw per-run rows.
- Ablation: W0/W1/W2 plus random-edge LLM refinement using the same number of calls as W2. This tests whether edge selection adds value.
- Sensitivity: a small predeclared query-budget sweep, such as 5%, 10%, and 20%, keeping the per-dataset cap. Use the primary detector to keep this small. Any further prompt/model variants must be agreed in the final protocol rather than silently deleted from the topic example.
- Significance: paired tests and effect sizes for a predeclared comparison. Five pairs have limited power. Do not add seeds or switch tests just to obtain a desired p-value.
- Case study: a few edges with changed weights and a graph figure, selected transparently and with both helpful and harmful cases where present.
- Downstream task: Independent Cascade simulations on both datasets, comparing top-degree seeds with community-based seeds from W1 and W2. Use k = 5, 10, 20, 50, p = 0.05, and 200 simulations per setting as the starting protocol from the guide's code.

Give each downstream strategy exactly k distinct nodes and use identical sampled diffusion conditions. The graph and diffusion probabilities must be the same across strategies. Otherwise the experiment would mix community quality with changes to the diffusion process.

The synthetic sanity check is an extra validation step. It does not replace real datasets. Citation subject labels are only a reference partition. We cannot claim observed real-world information diffusion from this simulation.

## Requirements mapped to evidence

Source: supplied PDF, printed pages 3, 5 to 12, 21 to 24, and 33 to 38. The PowerPoint adds career context, not submission requirements.

| Guide requirement | Evidence to prepare |
| --- | --- |
| Literature review: at least 15 papers, at least 10 from 2020 to 2026 | Literature matrix with papers actually opened, methods, datasets, metrics, and limitations |
| Gap supported by at least three references | A modest, evidence-backed gap paragraph; currently a research question, not a verified novelty claim |
| Base paper and experimental design | Approved paper/protocol, dataset list, metric definitions, baselines, pseudocode, complexity, risks |
| Reproducible implementation | GitHub source, uv lock and requirements export, Colab notebooks, one command per table/figure |
| Five seeds and complete experiments | Raw CSVs, configurations, cached LLM responses, tables, figures, ablation, sensitivity, significance |
| Downstream influence experiment | Results on two datasets with equal seed budgets |
| Checkpoint reporting | Short status notes and Overleaf/GitHub/Colab links sent by the student |
| Code/results freeze | Tagged `v1.0-results-freeze`; logged bug fixes and reruns afterward |
| Paper | Journal-formatted PDF with findings, limitations, references, declarations, code/data availability, AI disclosure |
| Instructor final package | Paper PDF, project links, similarity report, journal choice and backup |
| Journal communication | Instructor approval, one submission, acknowledgement containing manuscript ID |
| Viva | Slides, reproducible demonstration, ability to explain the method, code, comparisons, and limitations |

The five grading components are 6 marks each. Journal communication accounts for 3 marks, presentation/viva for 2, and checkpoint reporting for 1 within Communication. Acceptance is not required by the course guide. This does not remove the need for an honest manuscript appropriate to the venue.

## Phases and stopping points

Each phase ends with a concrete result that can be checked before the next phase starts. Only Phase 0 is being completed in this turn.

| Phase | Work | Done when |
| --- | --- | --- |
| 0. Research and setup | Verify relevant sources, select a practical proposal, create Desktop folder and uv files | Research notes and plan saved; core environment checked; unresolved academic choices explicit |
| 1. Approved design | Resolve solo/deadline/base-paper questions; complete the 15-paper matrix; fix the evaluation protocol | CP1 design package has the accepted base paper, scope, gap, pseudocode, exact metrics, and baselines |
| 2. Data and baselines | Verify archive access, original IDs, text coverage, preprocessing; build shared evaluation; run baseline detectors | Notebook 1 runs on both datasets; alignment checks and synthetic smoke test pass |
| 3. LLM method | Add embeddings, a bounded OpenRouter pilot, score validation, cache and usage logs | Notebook 2 runs W0/W1/W2 on one dataset with auditable costs and no manual result edits |
| 4. Full experiments | Complete the approved matrix, ablation, sensitivity, diffusion, resource measurements, and tests | All tables regenerate from saved raw outputs; clean-run check passes; CP2 freeze created |
| 5. Paper and assessment | Student writes the argument; assist with permitted polishing, formatting, reference checks, slides and viva preparation | Instructor receives the required package; approved journal submission evidence and viva materials are ready |

Unchanged guide dates: CP1 6 October; CP2 15 October; draft 5 November; final to instructor 10 November; feedback 12 November; journal submission by 15 November; viva 17 or 19 November 2026. As of 8 October, CP2 would be seven days away if those dates apply. The current deadline and available work hours are unknown, so this plan makes no time-to-completion guarantee.

## Phase 0 verification and limits

- The core uv environment installed successfully. NetworkX label propagation and igraph Leiden/Infomap ran on the built-in 34-node karate graph, with each node assigned exactly once. This verifies library operation, not the proposed method's quality. Optional notebook/text dependencies are locked but not installed or tested yet.
- A local Git repository is initialized, with no commits or remote. Secret files and local data/environment folders are ignored. `uv lock --check` passed. The complete requirements export is saved, but Colab compatibility remains to be tested.
- Primary-source technology and paper checks are recorded in [technical-research.md](technical-research.md) and [base-paper-research.md](base-paper-research.md).
- OpenRouter endpoint/pricing metadata was inspected without a key. Hosted inference has not been tested and no paid request was made.
- A full 15-paper literature review is still Phase 1 work. The research notes are a feasibility investigation, not a completed CP1 submission.
- Full raw data archives have not been loaded. PubMed text-to-node alignment is a specific item to prove in Phase 2.
- No project detection code, experiment results, notebook execution, Overleaf project, remote repository, or journal submission is claimed complete.

## Sources

Supplied project guide: `30mrks_cbca-215-social-networks.pdf`, titled *Minor Project Research Pipeline: Community Detection in Social Networks*, CSET347, 38 pages. Original at `/Users/rajatsharma/Desktop/30mrks_cbca-215-social-networks.pdf`.

Web sources supporting the choices and cautions:

- [TAPE official code and data release](https://github.com/XiaoxinHe/TAPE) and [paper](https://arxiv.org/html/2305.19523v3).
- [ComGPT author record and publication status](https://arxiv.org/abs/2408.06658).
- [uv project management](https://docs.astral.sh/uv/guides/projects/) and [notebooks](https://docs.astral.sh/uv/guides/integration/jupyter/).
- [OpenRouter integration](https://openrouter.ai/docs/quickstart), [Llama provider listing](https://openrouter.ai/meta-llama/llama-3.1-8b-instruct/providers), [routing](https://openrouter.ai/docs/guides/routing/provider-selection), and [usage records](https://openrouter.ai/docs/cookbook/administration/usage-accounting).
- [MiniLM model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), [NetworkX label propagation](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.community.label_propagation.asyn_lpa_communities.html), and [igraph community methods](https://python.igraph.org/en/main/api/igraph.community.html).
- [SciPy Wilcoxon test](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html).
- [IEEE TCSS author instructions](https://www.ieeesmc.org/publications/transactions-on-computational-social-systems/information-for-authors/).
