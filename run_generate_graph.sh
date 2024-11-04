#!/bin/bash

#SBATCH --job-name=generate_graph
#SBATCH --output=../outputs/slurm/generate_graph.txt
#SBATCH --partition=long
#SBATCH --mem=100G

python -u generate_graph.py --model 70b --simple --parser llama_parse --curated --task both


# python -u graphdocs2json.py --model 70b  --parser llama_parse --curated --task both

