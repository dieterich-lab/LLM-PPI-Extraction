#!/bin/bash

#SBATCH --job-name=generate_graph
#SBATCH --output=../outputs/slurm/generate_graph.txt
#SBATCH --mem=20G

python -u generate_graph.py --style 1 --target ppi_eval --doclevel --simple
python -u graphdocs2json.py --style 1 --target ppi_eval --doclevel --simple
python -u generate_graph.py --style 6 --target tf_eval --doclevel --simple
python -u graphdocs2json.py --style 6 --target tf_eval --doclevel --simple

# python -u generate_graph.py --style 1 --target ppi --doclevel --simple --curated
# python -u graphdocs2json.py --style 1 --target ppi --doclevel --simple --curated
# python -u generate_graph.py --style 6 --target tf --doclevel --simple --curated
# python -u graphdocs2json.py --style 6 --target tf --doclevel --simple --curated

# python -u generate_graph.py --style 1 --target lr_eval --doclevel --simple
# python -u graphdocs2json.py --style 1 --target lr_eval --doclevel --simple
# python -u generate_graph.py --style 2 --target lr_eval --doclevel --simple
# python -u graphdocs2json.py --style 2 --target lr_eval --doclevel --simple

# python -u generate_graph.py --style 3 --target lr_eval --doclevel --simple
# python -u graphdocs2json.py --style 3 --target lr_eval --doclevel --simple
# python -u generate_graph.py --style 4 --target lr_eval --doclevel --simple
# python -u graphdocs2json.py --style 4 --target lr_eval --doclevel --simple
# python -u generate_graph.py --style 5 --target lr_eval --doclevel --simple
# python -u graphdocs2json.py --style 5 --target lr_eval --doclevel --simple