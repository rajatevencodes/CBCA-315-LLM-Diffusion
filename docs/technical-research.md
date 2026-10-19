# Technical research

Checked 8 October 2026. Facts below are linked to primary sources. Choices labelled "decision" are project design decisions, not experimental findings.

## Python, uv, and notebooks

Decision: Python 3.12, already available locally, with a project environment managed by uv. Keep resolved dependencies in `uv.lock`; export `requirements.txt` for the guide and Colab. Use three short notebooks over shared source files. The uv documentation supports project environments, notebook use, and requirements exports. [Projects](https://docs.astral.sh/uv/guides/projects/), [Jupyter integration](https://docs.astral.sh/uv/guides/integration/jupyter/), [export command](https://docs.astral.sh/uv/reference/cli/#uv-export).

The core environment has graph, plotting, statistics, and HTTP libraries. Sentence Transformers and Jupyter are optional extras until the data phase. No custom model training is planned; PyTorch will be a dependency of the frozen sentence encoder, not a GNN training stack.

## Graph algorithms

NetworkX asynchronous label propagation accepts edge weights and a random seed. The algorithm is stochastic, so the five-seed evaluation in the guide remains necessary. [NetworkX API](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.community.label_propagation.asyn_lpa_communities.html).

igraph provides weighted Leiden and Infomap. Decision: use these together with NetworkX label propagation to keep the guide's three-detector example without adding separate cdlib, leidenalg, and Infomap packages. Set Leiden's objective explicitly to modularity; use a seeded Python random generator for igraph and a stable node ordering. Match these settings across weight variants. [igraph community API](https://python.igraph.org/en/main/api/igraph.community.html), [igraph graph API](https://python.igraph.org/en/main/api/igraph.Graph.html).

## Text embeddings

Decision: `sentence-transformers/all-MiniLM-L6-v2`, frozen, with normalized vectors. Its model card describes 384-dimensional sentence embeddings and a default truncation at 256 word pieces. Cache embeddings once per dataset and model revision. Report truncation rather than implying that every complete abstract was encoded. [Official model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).

Use original title/abstract text. Dataset labels, generated class predictions, and supervised TAPE features must not become input features for an experiment described as unsupervised. The separate base-paper note records why this matters and where to obtain raw text.

## OpenRouter and model selection

Decision: call the REST chat-completions endpoint using `requests`. This task needs a short score for a pair of texts, so an agent framework, tool calling, and conversation memory add no required capability. [Official quickstart](https://openrouter.ai/docs/quickstart).

Selected model: `meta-llama/llama-3.1-8b-instruct`. Selected provider for the initial pilot: `deepinfra/fp8`. The official endpoint listing returned input pricing of $0.02 and output pricing of $0.04 per million tokens on this check. An exact downloaded response is saved as `openrouter-llama-endpoints.json`. [Model page](https://openrouter.ai/meta-llama/llama-3.1-8b-instruct/providers), [endpoint API](https://openrouter.ai/api/v1/models/meta-llama/llama-3.1-8b-instruct/endpoints).

The general models endpoint returned different aggregate prices ($0.05/$0.08 per million) for the same model, while the endpoint-specific listing and model page showed the lower DeepInfra rate. Do not apply aggregate prices to a pinned provider or assume a listing guarantees a successful call. Recheck the exact endpoint before paid work. [Models API](https://openrouter.ai/api/v1/models).

This is a cheap, fixed Llama choice consistent with the guide's small-model example. It is not a claim that this is the cheapest model on the entire service. Picking a different or "latest" model automatically during experiments would change the method being evaluated.

Illustrative arithmetic, not measured usage: 10,000 calls with 800 input tokens and 16 output tokens each would cost $0.1664 at the above endpoint prices. Actual cost depends on token counts, retries, and routing. This estimate excludes any account funding fees or taxes. The selected main experiment caps unique edges at 2,000 per dataset and reuses responses, so it does not call the LLM again for each detector seed.

The routing API supports endpoint restrictions, disabling fallback, required parameters, and price ceilings. Decision: pin the endpoint for a recorded experiment, disable automatic fallback, use temperature 0, and stop on unavailability. A provider/model change creates a separately labelled experiment. Temperature 0 is not a promise of byte-identical hosted responses. [Provider routing](https://openrouter.ai/docs/guides/routing/provider-selection).

Current usage documentation says non-streaming responses include token counts and cost automatically. Store those values and the generation ID for every attempt; missing cost is unknown, not zero. [Usage accounting](https://openrouter.ai/docs/cookbook/administration/usage-accounting).

Decision: first pilot is at most 50 unique pairs. Proposed experiment spend ceiling is $1 before any extension. These settings are recorded in the proposal, but no spend limiter or API client is implemented yet. No API key has been requested or used in this phase. OpenRouter provides key-level limits and credit checks; the real client must enforce its own reservation/accounting as well. [Limits](https://openrouter.ai/docs/api_reference/limits).

## Cache and error handling to implement

Use one append-only JSONL cache, not a database. Identify a score using dataset/file hashes, stable node IDs, input text hashes, model, provider, prompt version, and parameters. Save the raw answer, validated score, tokens, cost, elapsed time, timestamp, and generation ID after each response. Reuse cached scores across the five community-detector seeds.

Accept only a whole integer from 0 to 10 after trimming whitespace. A timeout, refusal, malformed answer, or rate limit is not a score of zero. Retry a bounded number of times; retain the original embedding weight on terminal failure and report the number of such fallbacks. A run with incomplete data is labelled accordingly. Exact validation avoids the guide's starter-code behaviour of reading the first digits from any answer.

## Evaluation checks that affect the research

Use arithmetic NMI explicitly and ARI without altering predicted labels. For clustering ACC and macro-F1, compute a one-to-one Hungarian assignment and handle different numbers of true and predicted groups without changing the detected partition. Report unmatched clusters, singleton counts, and community counts. [NMI](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.normalized_mutual_info_score.html), [ARI](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html), [linear assignment](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html).

Decision: compare structural modularity on the same original unweighted graph for every method. Modularity depends on the graph's weights, so comparing values measured on three different reweighted graphs would confound the comparison. Method-specific weighted modularity can be supplementary and must be labelled. [NetworkX modularity definition](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.community.quality.modularity.html).

The guide requests five seeds and paired significance testing. With five distinct nonzero paired differences, the smallest exact two-sided signed-rank p-value is 2/32 = 0.0625. Thus p < 0.05 is impossible in that case. This is a mathematical consequence of enumerating the 32 sign assignments, not a reason to change the test after seeing the result. Report the honest p-value and effect size. Detector randomness on a fixed graph/cache does not measure uncertainty over new datasets or new LLM responses. [SciPy Wilcoxon definition and exact-test conditions](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html).

For the required downstream experiment, choose exactly k distinct seed nodes under each strategy and use the same original graph and transmission probability for all methods. Reuse the same sampled live-edge graphs across strategies for a paired simulation. Otherwise, reweighting the diffusion process itself can manufacture an apparent benefit for the proposed detector. The course guide requires two datasets, k in {5,10,20,50}, and the Independent Cascade model (page 12); its example uses p=0.05 and 200 runs.

## Paper and venue expectations

Proposed first target: IEEE Transactions on Computational Social Systems, one of the guide's three main choices. Its scope covers computational analysis of social systems and network structure. Its author page specifies IEEE two-column format, regular papers normally around ten pages or shorter, and a 100 to 250 word abstract. A planned 150 to 250 word abstract also fits the guide. The page links the submission system and template. This establishes fit and format, not likely acceptance. [Journal scope](https://www.ieeesmc.org/publications/transactions-on-computational-social-systems/), [author instructions](https://www.ieeesmc.org/publications/transactions-on-computational-social-systems/information-for-authors/).

Proposed backup: Expert Systems with Applications, already on the guide's primary list. Its current detailed author page was not accessible during this check, so its exact format and charges remain to be verified before selecting it. No payment-free publication promise is made. [Official journal listing](https://shop.elsevier.com/journals/expert-systems-with-applications/0957-4174).

The guide's statement that all listed publishers offer a subscription route is not safe as a blanket rule. Social Network Analysis and Mining currently identifies its publishing model as open access. Do not select it as a free backup without checking the actual fee/waiver route. [Journal home](https://link.springer.com/journal/13278).

The supplied guide restricts generative AI in paper writing to language polishing where permitted. Keep that course rule. Elsevier's current policy describes broader permitted support with human verification and disclosure, and asks for research-method AI to be detailed in Methods. A publisher policy does not cancel a stricter course instruction. Record AI assistance and never treat it as a source for unverified facts or numerical results. [Elsevier policy](https://www.elsevier.com/about/policies-and-standards/generative-ai-policies-for-journals).
