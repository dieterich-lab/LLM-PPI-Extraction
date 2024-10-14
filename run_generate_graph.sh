#!/bin/bash

#SBATCH --job-name=generate_graph
#SBATCH --output=../outputs/slurm/generate_graph.txt
#SBATCH --partition=long
#SBATCH --mem=100G

python -u generate_graph.py --model 8b  --parser llama_parse
python -u generate_graph.py --model 8b --simple --parser llama_parse
python -u generate_graph.py --model 70b  --parser llama_parse
python -u generate_graph.py --model 70b --simple --parser llama_parse
python -u generate_graph.py --model 8b  --parser marker
python -u generate_graph.py --model 8b --simple --parser marker
python -u generate_graph.py --model 70b  --parser marker
python -u generate_graph.py --model 70b --simple --parser marker
