#!/bin/bash

#SBATCH --job-name=graphdocs2json
#SBATCH --output=../outputs/slurm/graphdocs2json.txt
#SBATCH --partition=long
#SBATCH --mem=100G


python -u graphdocs2json.py --model 70b  --parser llama_parse --curated --task ppi
python -u graphdocs2json.py --model 70b  --parser llama_parse --curated --task tf
python -u graphdocs2json.py --model 70b --simple --parser llama_parse --curated --task ppi
python -u graphdocs2json.py --model 70b --simple --parser llama_parse --curated --task tf
python -u graphdocs2json.py --model biollm  --parser llama_parse --curated --task ppi
python -u graphdocs2json.py --model biollm  --parser llama_parse --curated --task tf
python -u graphdocs2json.py --model biollm --simple --parser llama_parse --curated --task ppi
python -u graphdocs2json.py --model biollm --simple --parser llama_parse --curated --task tf

