#!/bin/bash

#SBATCH --job-name=generate_graph
#SBATCH --output=../outputs/slurm/generate_graph.txt
#SBATCH --partition=long
#SBATCH --mem=100G



python -u generate_graph.py --style 1 --target ppi_eval --doclevel --simple
python -u graphdocs2json.py --style 1 --target ppi_eval --doclevel --simple
python -u generate_graph.py --style 6 --target tf_eval --doclevel --simple
python -u graphdocs2json.py --style 6 --target tf_eval --doclevel --simple

python -u generate_graph.py --style 1 --target lr --doclevel --simple
python -u graphdocs2json.py --style 1 --target lr --doclevel --simple
python -u generate_graph.py --style 2 --target lr --doclevel --simple
python -u graphdocs2json.py --style 2 --target lr --doclevel --simple
python -u generate_graph.py --style 3 --target lr --doclevel --simple
python -u graphdocs2json.py --style 3 --target lr --doclevel --simple
python -u generate_graph.py --style 4 --target lr --doclevel --simple
python -u graphdocs2json.py --style 4 --target lr --doclevel --simple
python -u generate_graph.py --style 5 --target lr --doclevel --simple
python -u graphdocs2json.py --style 5 --target lr --doclevel --simple