#!/bin/bash

#SBATCH --job-name=samples_8b
#SBATCH --output=../outputs/slurm/generate_graph_llama31_8b_samples.txt
#SBATCH --partition=long
#SBATCH --mem=100G


python -u generate_graph.py --samples --parser llama_parse --model 8b
python -u generate_graph.py --samples --parser marker --model 8b
