#!/bin/bash

#SBATCH --job-name=generate_graph_samples
#SBATCH --output=../outputs/slurm/generate_graph_samples.txt
#SBATCH --partition=long
#SBATCH --mem=100G

python -u generate_graph.py --parser llama_parse
python -u generate_graph.py --parser marker
