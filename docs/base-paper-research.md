# Base paper and dataset research

Checked 8 October 2026. This note records primary-source checks and a recommendation. No final base paper, dataset exception, or instructor approval has been assumed. No dataset archive or baseline was run during this review.

## Recommendation

Use the guide's Section 9 semantic edge weighting project as the proposed design: raw paper text, sentence embeddings for every edge, an LLM for a capped set of uncertain edges, then community detection. Cora and PubMed are the practical proposed datasets. The research question is whether selective LLM scoring changes community quality enough to justify its measured cost.

The formal base-paper decision remains open. The guide's suggested papers do not supply a directly matching, two-dataset, unsupervised clustering protocol. TAPE is a useful data reference. ComGPT is useful related work. Calling them a directly reproduced clustering baseline would be inaccurate.

The instructor should approve the following concrete adaptation before the final experiment protocol is frozen:

> Follow the Section 9 semantic diffusion design on Cora and PubMed using the TAPE raw text release. Use unsupervised community detection metrics and freshly run comparable baselines. Treat TAPE as the data reference and ComGPT as related work, rather than importing their published scores into the main comparison table. Please confirm the primary base paper and the permitted reduced dataset and baseline scope for this solo project.

This is required by the supplied course guide, not by a research skill. Section 5 requires the base paper's datasets, metrics, and primary method; Section 9 calls its own default plan illustrative and subject to CP1 approval. Its large-dataset substitution clause does not plainly authorize dropping arbitrary datasets. Source: supplied PDF, printed pages 9 and 21 to 24.

## Candidate papers checked

| Candidate | Verified status and experiment | Fit for this project |
|---|---|---|
| Xiaoxin He, Xavier Bresson, Thomas Laurent, Adam Perold, Yann LeCun, Bryan Hooi. *Harnessing Explanations: LLM-to-LM Interpreter for Enhanced Text-Attributed Graph Representation Learning*. ICLR 2024. | TAPE uses five datasets: Cora, PubMed, ogbn-arxiv, ogbn-products subset, tape-arxiv23. The task is node classification and the reported main metric is classification accuracy. | Use its raw text release. Its supervised training and published accuracy are not an unsupervised clustering baseline. |
| Li Ni, Haowen Shen, Lin Mu, Yiwen Zhang, Wenjian Luo. *ComGPT: Detecting Local Community Structure with Large Language Models*. | Author arXiv record now lists IEEE Transactions on Computational Social Systems 2026 and DOI 10.1109/TCSS.2025.3634341. Five datasets: Dolphins, Football, Polbooks, Amazon, DBLP. Metrics: local-community Jaccard and Fscore. | The guide's preprint label is outdated. This finds a community around a supplied seed; our proposed method partitions an entire text graph. Different task and datasets. |
| Ekta Gujral, Apurva Sinha. *LLMs Between the Nodes: Community Discovery Beyond Vectors*. 2025. | CommLLM has a publisher proceedings version, DOI 10.5121/csit.2025.151412. Six datasets: Karate Club, Football, WebKB, Terrorist Attacks, Cora, CiteSeer. Metrics: NMI, ARI, variation of information, purity. | Whole graph adjacency is converted to text for GPT-4o. Its large prompt and partial output handling are a poor fit for a cheap edge-scoring experiment. The course's major-conference/reputed-venue criterion needs instructor judgment. |
| Soumita Das, Anupam Biswas, Akrati Saxena. *Modeling homophily fusion on cascading for overlapping community detection in social networks*. ESWA 307, 130963, 2026. | Sangam evaluates 12 real networks and 71 LFR networks using extended modularity, overlapping modularity, and NF1. DOI 10.1016/j.eswa.2025.130963. | Relevant diffusion background, but overlapping membership and experiment scale are unlike the proposed partitioning project. Public implementation and complete dataset list were not verified from accessible primary material. |

Sources: [TAPE paper](https://arxiv.org/html/2305.19523v3), [TAPE official repository](https://github.com/XiaoxinHe/TAPE), [ComGPT author record](https://arxiv.org/abs/2408.06658), [ComGPT full paper](https://arxiv.org/html/2408.06658v3), [CommLLM full paper](https://arxiv.org/html/2507.22955v1), [CommLLM publisher PDF](https://aircconline.com/csit/papers/vol15/csit151412.pdf), [Sangam publisher page](https://www.sciencedirect.com/science/article/abs/pii/S0957417425045786).

ComGPT's official repository contains code and all five graph folders. The paper names `gpt-3.5-turbo-0125`. Replacing this with a Llama model would be an adaptation and must be labelled as such. No root license file was present in the inspected tree. The IEEE DOI did not load in this tool, so the journal status above is grounded in the authors' updated arXiv record, not a successfully opened IEEE page. [Official code](https://github.com/cassius0301/ComGPT).

## Closest additional candidate

Xinchuang Zhou, Lingtao Su, Xiangju Li, Zhongying Zhao, Chao Li. *Community detection based on unsupervised attributed network embedding*. Expert Systems with Applications 213, 118937, 2023. DOI 10.1016/j.eswa.2022.118937.

CDBNE is a closer task match. The publisher identifies unsupervised community detection, three citation datasets (Cora, CiteSeer, PubMed), and an official implementation. It combines a graph attention autoencoder, modularity objective, and self-training clustering. It is a fallback candidate, not a confirmed easy baseline. [Publisher](https://www.sciencedirect.com/science/article/pii/S0957417422019558).

Inspection of the official code found practical problems to resolve before choosing it:

- `run.py` expects a pretrained checkpoint, uses dense graph operations, defaults to 500 epochs, and creates a t-SNE plot inside the training loop.
- Its evaluation helper changes some predicted labels when predicted and true class counts differ. That must not be copied into our general detector evaluation.
- Its reporting prints the maximum of each metric over epochs. A comparison needs a fixed, defensible checkpoint selection rule, not selection using final evaluation labels.
- The inspected root has no setup README or license file. A usable environment and redistribution terms need verification.

Its code computes matched ACC, macro-F1, arithmetic NMI, and ARI. The full publisher metric definitions were not accessible in this review, so code output alone does not establish the exact paper protocol. CiteSeer raw text with matching node IDs would also be needed. [Training code](https://github.com/xidizxc/CDBNE/blob/master/run.py), [evaluation code](https://github.com/xidizxc/CDBNE/blob/master/evaluation.py), [repository](https://github.com/xidizxc/CDBNE).

Other checks did not establish a simpler ready base. SEComm is UAI 2021, outside the guide's 2022 to 2026 range. DGCLUSTER is AAAI 2024 with public code, but adds GNN training and a broader experiment suite. A 2025 Neurocomputing paper on text-attributed community detection explicitly defines its task as supervised node classification on four heterogeneous graph datasets. [SEComm proceedings](https://proceedings.mlr.press/v161/bandyopadhyay21a/bandyopadhyay21a.pdf), [DGCLUSTER proceedings](https://ojs.aaai.org/index.php/AAAI/article/view/28983), [DGCLUSTER code](https://github.com/pyrobits/DGCluster), [Neurocomputing paper](https://www.sciencedirect.com/science/article/pii/S0925231225012457).

## Raw text availability and alignment

TAPE's official README links original Cora and PubMed text archives separately from its GPT responses and pretrained features. Use the original text. Their Drive links were verified in the README, but successful download and complete text coverage remain Phase 2 checks. [Download instructions](https://github.com/XiaoxinHe/TAPE#1-download-tag-datasets).

The inspected TAPE source was revision `d9881f7e277823d0c3a56851c6ccd0412adc773f`.

For Cora, `parse_cora` reads paper IDs from `cora.content` and maps `cora.cites` through those IDs. The raw text loader looks up each paper ID in the `papers` index and reads the matching extraction file. It then returns text in that same node order. Preserve this ID join rather than pairing raw text with an independently loaded graph by row position. Validate missing title/abstract fields: the original loop does not reset those variables for each paper. [Cora source](https://github.com/XiaoxinHe/TAPE/blob/d9881f7e277823d0c3a56851c6ccd0412adc773f/core/data_utils/load_cora.py).

For PubMed, the parser builds node order from `Pubmed-Diabetes.NODE.paper.tab` and maps citation endpoints with original paper IDs. However, the text loader walks JSON title/abstract rows without visibly joining them back to the graph IDs. Check archive IDs and prove alignment before trusting it. Do not assume that equal row counts prove correct alignment. Avoid copying its dense 19,717 by 19,717 adjacency allocation into a simple edge-list loader. [PubMed source](https://github.com/XiaoxinHe/TAPE/blob/d9881f7e277823d0c3a56851c6ccd0412adc773f/core/data_utils/load_pubmed.py).

TAPE's code has an MIT license. This does not establish that every upstream abstract or dataset is covered by the same license. Preserve code attribution. Keep download URLs, archive checksums, original IDs, and upstream terms in a data manifest. Distribute download instructions before deciding whether raw abstracts may be redistributed. [Code license](https://github.com/XiaoxinHe/TAPE/blob/main/LICENSE).

## Protocol implications

These are design recommendations, not claims that an experiment has passed:

- Use raw text and frozen general-purpose sentence embeddings. Do not silently use TAPE features trained on the labels being used for our evaluation.
- Citation subject labels are a reference partition, not proof of real social groups. Explain that limitation.
- A semantic score is an affinity weight, not a measured probability that a real person will adopt information. Citation graphs have no observed adoption outcomes in this design.
- Evaluate all methods on exactly the same graph and node IDs. Report self-loop removal, undirected conversion, duplicate-edge treatment, and missing-text handling.
- Fix the LLM query policy before scoring the final evaluation labels. A cosine middle band is a heuristic for selecting edges, not a calibrated uncertainty estimate.
- Run any chosen baselines ourselves under the same protocol. Do not import TAPE classification, ComGPT local-retrieval, or Sangam overlapping-community numbers into a global partitioning results table.
- Keep a capped pilot, cached LLM answers, and a no-LLM baseline. A result showing no benefit from LLM calls is still a valid result if reported honestly.

The unresolved issues are the instructor's accepted base-paper protocol, approved reduced scope, archive access and ID alignment, and the runtime of the selected comparable baseline. Setup and synthetic smoke checks can proceed while these are resolved. Final research claims and the full paid experiment should wait for a fixed protocol.
