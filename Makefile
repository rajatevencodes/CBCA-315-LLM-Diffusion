# Semantic Diffusion - one command per step. Run `make help` to list them.
RUN = uv run python src/run_pipeline.py

.PHONY: help init data weights pilot llm experiments diffusion charts all requirements notebook

help:          ## list commands
	@grep -E '^[a-z]+:.*##' Makefile | awk -F':.*## ' '{printf "  make %-12s %s\n", $$1, $$2}'

init:          ## install Python 3.13 + packages, create .env
	uv sync
	@test -f .env || (cp .env.example .env && echo "Created .env - paste your OPENROUTER_API_KEY into it")

data:          ## step 1: download + align Cora and PubMed
	$(RUN) --step data

weights:       ## step 2: text embeddings -> W0 / W1 weights
	$(RUN) --step weights

pilot:         ## step 3a: try the LLM on 50 edges (costs < $$0.01)
	$(RUN) --step pilot

llm:           ## step 3b: LLM scores for all selected edges (cached)
	$(RUN) --step llm

experiments:   ## step 4: community detection runs -> outputs/runs.csv
	$(RUN) --step experiments

diffusion:     ## step 5: influence-spread downstream task
	$(RUN) --step diffusion

charts:        ## step 6: tables, tests, charts -> outputs/
	$(RUN) --step charts

all:           ## run every step in order
	$(RUN)

requirements:  ## refresh requirements.txt (used by Google Colab)
	uv export --no-hashes --no-dev --no-emit-package torch --format requirements-txt -o requirements.txt

notebook:      ## open Jupyter with the project environment
	uv run --with jupyterlab jupyter lab notebooks/
