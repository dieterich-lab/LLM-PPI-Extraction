#!/bin/bash

#SBATCH --job-name=filter_cardiac
#SBATCH --output=../../outputs/slurm/filter_cardiac_papers.log
#SBATCH --partition=medium
#SBATCH --mem=5G

cd ../ &&
. ~/.venvs/test_linda/bin/activate

export CUDA_VISIBLE_DEVICES=1

MODEL="${MODEL:-llama31}"
RESULTS_JSONL="/prj/LINDA_LLM/outputs/cardiac_filter/ppi_results_${MODEL}.jsonl"

poetry run python -m cardiac_filtering.filter_cardiac_papers \
	--cardioprior-ppi \
	--model "$MODEL" \
	--node g4 \
	--port 34 \
	--loglevel off \
	--max-chars 40000 \
	--output-jsonl "$RESULTS_JSONL"
