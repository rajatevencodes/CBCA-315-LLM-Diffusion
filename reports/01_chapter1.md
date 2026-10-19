# Chapter 1. Problem analysis and introduction

## 1.1 Background of the study

A citation network is a social network of ideas. Each research paper is a node, and each citation is an edge. Papers on the same subject cite each other more often, so groups of densely connected papers, called *communities*, usually match research areas. Finding these communities automatically is called *community detection* [4].

Many community detection methods work by simulating how information spreads. Label propagation lets every node repeatedly adopt the most common label among its neighbours until labels settle [2]. Infomap follows a random walker and finds groups where the walker stays for a long time [3]. Leiden moves nodes between groups to maximise modularity, a score that compares the number of edges inside groups with the number expected by chance [5]. All three can use edge *weights*: a heavier edge pulls its two endpoints into the same community more strongly.

By default every citation gets the same weight. That ignores the most informative part of a paper, its text. A citation between two papers on the same narrow topic is a stronger signal than a citation to a general method paper from a different field. Text embeddings such as Sentence-BERT turn a title and abstract into a vector, and the cosine similarity of two vectors measures how close two texts are [6].

Large language models (LLMs) can judge the relation between two texts more carefully than an embedding [11], [12]. They are also slower and cost money for every call. Recent work uses LLMs on graphs as feature enhancers, predictors or annotators [11], [12], [13], [14], [17], and two recent studies use LLMs directly for community detection [19], [20]. Both rely on large proprietary models and do not report how much the LLM calls cost compared with cheap embeddings.

This project follows the Topic 2 plan of the course guide: weight every edge by text similarity, ask a small open LLM to re-judge only the ambiguous edges, and measure whether the extra quality is worth the extra cost.

## 1.2 Problem statement

Structure-only community detection treats every citation as equally meaningful. Text embeddings add content, but they are noisy for edges whose similarity is in the middle range. An LLM may judge those middle edges better, but calling it for every edge in a large network is slow and costly, and nobody has measured whether a *small* number of calls to a *small* model changes the result at all.

The research question is:

> *Does re-scoring a limited number of edge weights with a small LLM improve community detection in text-attributed citation networks enough to justify its API cost, compared with text-embedding weights alone?*

Five practical questions follow:

1. Do text-similarity weights (W1) find better communities than structure alone (W0)?
2. Does replacing 10% of the weights with LLM scores (W2) improve on W1, and for which detector?
3. Is choosing the *ambiguous* edges better than spending the same number of calls on random edges?
4. How do quality and cost change with the budget (5, 10, 20% of edges), the prompt wording and the model size?
5. Do communities from W2 choose better seed nodes for spreading information under the Independent Cascade model [1]?

## 1.3 Objectives

Each objective has a measurable target. Chapter 4 reports the results.

1. Load Cora and PubMed with their **original** titles and abstracts, join text to nodes by paper ID, and record checksums, node and edge counts and text coverage in `outputs/data_manifest.json`.
2. Show on a synthetic graph with 4 planted communities that the pipeline recovers the planted groups (target: NMI ≥ 0.9 with text weights).
3. Build three weight variants (W0, W1, W2) on the same graph. W2 calls the LLM for at most 10% of edges and never turns a failed call into a score.
4. Run three detectors (label propagation, Leiden, Infomap) with 5 seeds each and report NMI, ARI, ACC, macro-F1 and modularity as mean ± standard deviation.
5. Log every LLM call with tokens, cost and latency, enforce a $1.00 spending ceiling, and report the cost of each variant.
6. Run the ablation (random edges, shifted bands, a second prompt, a larger 70B model) and the budget sensitivity sweep (5, 10, 20%).
7. Test W2 against W1, W0 and random-edge W2 with paired Wilcoxon signed-rank tests and report effect sizes.
8. Compare influence spread for k ∈ {5, 10, 20, 50} seeds on both datasets, with identical diffusion conditions for every strategy.
9. Regenerate every table and chart from one command (`make all`).

## 1.4 Literature review

The review covers 21 papers: 7 foundational works and 14 from 2020 to 2026. Every entry was checked against its DOI or official proceedings page.

### 1.4.1 Diffusion-based community detection and influence spread

Kempe, Kleinberg and Tardos defined the Independent Cascade model, in which each newly activated node gets one chance to activate each neighbour with probability p, and showed that choosing seed nodes to maximise spread is NP-hard but approximable by a greedy algorithm [1]. Raghavan, Albert and Kumara introduced label propagation, which runs in near-linear time but gives different results for different random orders [2]. Rosvall and Bergstrom's Infomap finds communities by compressing the description of a random walk [3]. Fortunato's survey defines the field and its evaluation problems [4]. Traag, Waltman and van Eck showed that Louvain can produce badly connected communities and proposed Leiden, which guarantees connected communities and runs faster [5]. Meena, Singh and Singh choose seeds that reach as many communities as possible in dynamic networks, using bridge nodes between communities [18]. Das, Biswas and Saxena fuse homophily into a cascading process to find overlapping communities and evaluate on 12 real networks and 71 synthetic LFR networks [21]. None of these methods use the text of the nodes when deciding how information flows along an edge.

### 1.4.2 Attributed and deep community detection

Su et al. survey deep learning for community detection, covering graph neural networks, autoencoders and deep matrix factorisation, and summarise the benchmark datasets and metrics [8]. CDBNE combines a graph attention autoencoder, a modularity term and self-training, and reports ACC, NMI, ARI and F1 on Cora, CiteSeer and PubMed [9]. DGCluster trains a graph neural network to maximise modularity without fixing the number of clusters [10]. These methods need neural training and use bag-of-words features rather than the raw text.

### 1.4.3 LLMs and text-attributed graphs

Sen et al. released Cora and PubMed as benchmark citation graphs [7]. Jin et al. [11] and Li et al. [17] survey LLMs on graphs and group them by the role the LLM plays: enhancer, predictor or aligner. Chen et al. compare LLMs as feature enhancers and as direct predictors on Cora, PubMed and OGB datasets [12]. TAPE prompts an LLM to explain each paper's likely category and uses the explanation as an extra feature; its release provides the original Cora and PubMed text used here [13]. LLM-GNN uses an LLM to label a small, carefully chosen set of nodes and trains a GNN on them, reaching 74.9% accuracy on ogbn-products for less than one US dollar [14]. This is the closest idea to ours, a *budgeted* use of the LLM, but for classification, not community detection. Fatemi et al. show that how a graph is written as text changes LLM accuracy a lot [15]. GraphGPT aligns graph structure with an LLM through instruction tuning [16].

### 1.4.4 LLM-based community detection

ComGPT uses GPT-3.5 to grow a *local* community around one given seed node and reports Jaccard and F-score on five networks [19]. CommLLM writes the graph into a GPT-4o prompt and asks for the communities, with NMI, ARI, variation of information and purity on six datasets including Cora [20]. Both depend on large proprietary models, send large prompts and do not compare against cheap embedding weights at equal cost.

### 1.4.5 Literature matrix

**Table 1.1: Literature matrix (course guide, Appendix A)**

| # | Citation | Problem | Method (1-2 lines) | Datasets | Metrics | Key result | Limitation / future work |
|---|---|---|---|---|---|---|---|
| 1 | Kempe et al., 2003, KDD [1] | Influence maximisation | Independent Cascade and Linear Threshold models; greedy seed selection | Co-authorship networks | Expected spread | Greedy is within 1 - 1/e of optimal | Ignores content; costly simulations |
| 2 | Raghavan et al., 2007, Phys. Rev. E [2] | Fast community detection | Label propagation | Karate club, college football and larger real networks | Modularity, runtime | Near-linear time | Random, unstable results; no content |
| 3 | Rosvall & Bergstrom, 2008, PNAS [3] | Community detection by flow | Infomap: compress a random walk | Citation networks of journals | Description length | Finds flow-based modules | Edge weights must be given |
| 4 | Fortunato, 2010, Phys. Rep. [4] | Survey | Taxonomy of methods and tests | Benchmarks | NMI, modularity | Defines evaluation practice | Pre-LLM |
| 5 | Traag et al., 2019, Sci. Rep. [5] | Badly connected communities | Leiden refinement step | Real and LFR graphs | Modularity, connectivity, time | Connected communities, faster than Louvain | Structure only |
| 6 | Reimers & Gurevych, 2019, EMNLP [6] | Sentence embeddings | Siamese BERT (Sentence-BERT) | STS, NLI | Spearman correlation | Fast semantic similarity by cosine | General-purpose; misses domain nuance |
| 7 | Sen et al., 2008, AI Mag. [7] | Collective classification | Benchmarks and methods | Cora, CiteSeer, PubMed | Accuracy | Standard citation benchmarks | Bag-of-words features only |
| 8 | Su et al., 2024, IEEE TNNLS [8] | Survey of deep community detection | Taxonomy: DNNs (GCN, GAT, GAN, autoencoder), deep NMF, deep sparse filtering | Benchmark collections | NMI, ARI, ACC, F1 | Map of deep methods, datasets and code | Open challenges listed, incl. attributed networks |
| 9 | Zhou et al., 2023, ESWA [9] | Unsupervised attributed CD | Graph attention autoencoder + modularity + self-training (CDBNE) | Cora, CiteSeer, PubMed | ACC, NMI, ARI, F1 | Beats earlier embedding methods | Neural training; bag-of-words input |
| 10 | Bhowmick et al., 2024, AAAI [10] | Attributed graph clustering | GNN trained to maximise modularity, linear in graph size (DGCluster) | Several real graphs of varying size | Several cluster-quality metrics | No need to fix the number of clusters | Neural training needed |
| 11 | Jin et al., 2024, IEEE TKDE [11] | Survey of LLMs on graphs | Roles: predictor, encoder, aligner | Many | Various | Map of the field | Cost and community tasks rarely covered |
| 12 | Chen et al., 2024, SIGKDD Explor. [12] | LLMs for node classification | LLM as enhancer vs predictor | Cora, PubMed, ogbn-arxiv, ogbn-products | Accuracy | Enhancers help; predictors are mixed | Inference cost |
| 13 | He et al., 2024, ICLR [13] | Text-attributed graph learning | LLM explanations as features (TAPE) | Cora, PubMed, ogbn-arxiv, ogbn-products, arxiv-2023 | Accuracy | State-of-the-art accuracy | Supervised; one LLM call per node |
| 14 | Chen et al., 2024, ICLR [14] | Learning without labels | LLM annotates a selected node subset, GNN learns (LLM-GNN) | Cora, CiteSeer, PubMed, WikiCS, OGB | Accuracy, cost | 74.9% on ogbn-products for < $1 | Classification, not communities |
| 15 | Fatemi et al., 2024, ICLR [15] | Graphs as LLM input | Compare text encodings of graphs | GraphQA benchmark | Accuracy | Encoding choice changes accuracy greatly | Small synthetic graphs |
| 16 | Tang et al., 2024, SIGIR [16] | Graph-aware LLM | Graph instruction tuning (GraphGPT) | Cora, PubMed, arXiv | Accuracy, macro-F1 | Strong zero-shot transfer | Fine-tuning cost |
| 17 | Li et al., 2024, IJCAI [17] | Survey of graph + LLM | Enhancer / predictor / alignment taxonomy | Many | Various | Taxonomy and open problems | Efficiency open |
| 18 | Meena et al., 2024, ACM TWeb [18] | Diversified influence maximisation | Objective = communities reached, using bridge nodes; new diffusion model (DCDIMB) | Real and synthetic dynamic networks | Communities activated, spread | Reaches more communities than benchmarks | Structure only; no node content |
| 19 | Ni et al., 2026, IEEE TCSS [19] | Local community detection | GPT-3.5 expands a community from a seed (ComGPT) | Dolphins, Football, Polbooks, Amazon, DBLP | Jaccard, F-score | Beats local baselines | Local only; proprietary LLM; cost not studied |
| 20 | Gujral & Sinha, 2025, CS & IT [20] | Global community detection | Whole graph in a GPT-4o prompt (CommLLM) | Karate, Football, WebKB, Terrorist Attacks, Cora, CiteSeer | NMI, ARI, VI, purity | Competitive on small graphs | Prompt size limits scale; cost |
| 21 | Das et al., 2026, ESWA [21] | Overlapping CD via cascades | Homophily fusion on cascading | 12 real + 71 LFR networks | Extended / overlapping modularity, NF1 | Better overlapping partitions | No text content |

## 1.5 Research gap

Three limitations recur across the matrix:

1. **Content-blind diffusion.** Diffusion-based detectors treat every edge as equally likely to carry information [2], [3], [18], [21].
2. **Cost is not measured.** LLM-based community methods use large proprietary models and do not report cost against a cheap embedding baseline [19], [20]. Surveys list efficiency as an open problem [11], [17].
3. **Targeted LLM use is unexplored for communities.** Budgeted LLM queries work for node classification [14], but querying the LLM only for uncertain edges under a fixed budget has not been tested for community detection.

The gap addressed here is the third, measured with the second in mind. We do not claim that LLM edge weighting is new; the course guide proposes it.

## 1.6 Contributions

1. A reproducible pipeline that aligns original text to Cora and PubMed nodes by paper ID. It fixes two alignment problems in the reference loader.
2. A budgeted edge re-weighting method: a small open LLM (Llama 3.1 8B) re-scores at most 10% of edges, chosen from an ambiguity band, with every call cached and costed.
3. A controlled comparison of W0, W1 and W2 across three detectors, five seeds, a random-edge control, band, prompt, model and budget ablations, and a downstream influence-spread task.
4. An honest answer to the cost question, including where the LLM does not help.

## 1.7 Organisation of the report

Chapter 2 describes the design: datasets, method, pseudocode, metrics and protocol. Chapter 3 describes the implementation. Chapter 4 presents and discusses the results. Chapter 5 concludes with limitations, future work, declarations and the full reference list. Appendix A gives the exact prompts and extra tables.
