#!/bin/bash

#SBATCH --job-name=generate_graph_both
#SBATCH --output=../outputs/slurm/generate_graph_both.txt
#SBATCH --partition=medium
#SBATCH --mem=100G

python -u generate_graph.py --model biollm  --parser llama_parse --curated --task both
python -u generate_graph.py --model biollm --simple --parser llama_parse --curated --task both
python -u generate_graph.py --model 70b  --parser llama_parse --curated --task both
python -u generate_graph.py --model 70b --simple --parser llama_parse --curated --task both
