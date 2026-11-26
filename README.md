# Semantic Diffusion: Budgeted LLM Edge Weights for Community Detection

This is a college project: CBCA315 Minor Project (Topic 2), BCA, Bennett University.
Student: Rajat Sharma (S24BCAU0134). Guide: Dr. Soumita Das.

## Links

* GitHub: https://github.com/rajatevencodes/CBCA-315-LLM-Diffusion
* Colab: https://colab.research.google.com/github/rajatevencodes/CBCA-315-LLM-Diffusion/blob/main/notebooks/semantic_diffusion.ipynb
* Overleaf: https://www.overleaf.com/project/6b07ab9563d33d0cff55f0f1/share#253bdeaea3eceaa527a947b5849e0ba4ffd85f2b1a23f0e3

## What it does

Each paper is a node and each citation is an edge. Every edge gets a weight from how similar the two papers' texts are. A small LLM (Llama 3.1 8B) re-scores the 10% most ambiguous edges. Label propagation, Leiden and Infomap then find communities, and we check if the LLM helps enough to justify its cost. It did not: see `outputs/tables/`.

## Run it

```bash
make init   # install, then put OPENROUTER_API_KEY in .env
make all    # every step; saved LLM answers are reused, so no new cost
```

Step by step guide: [setup.md](setup.md).
