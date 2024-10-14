#!/bin/bash

#SBATCH --job-name=generate_graph
#SBATCH --output=../outputs/slurm/generate_graph_llama3_1.txt
#SBATCH --partition=long
#SBATCH --mem=100G

python -u generate_graph.py
