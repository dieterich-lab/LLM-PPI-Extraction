#!/bin/bash

#SBATCH --job-name=generate_graph
#SBATCH --output=../outputs/slurm/generate_graph_llama31_8b_samples_simple.txt
#SBATCH --partition=long
#SBATCH --mem=100G


python -u generate_graph.py --parser llama_parse
python -u generate_graph.py --parser marker
