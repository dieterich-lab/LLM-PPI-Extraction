#!/bin/bash

#SBATCH --job-name=generate_graph_405b
#SBATCH --output=../outputs/slurm/generate_graph_405b_curated.txt
#SBATCH --mem=20G




# python -u generate_graph.py --style 1 --target ppi --doclevel --simple --model 405b --nebius --curated
# python -u graphdocs2json.py --style 1 --target ppi --doclevel --simple --model 405b --nebius --curated
python -u generate_graph.py --style 6 --target tf --doclevel --simple --model 405b --nebius --curated
python -u graphdocs2json.py --style 6 --target tf --doclevel --simple --model 405b --nebius --curated 

# python -u generate_graph.py --style 1 --target tf --doclevel --simple --model 405b --nebius --curated
# python -u generate_graph.py --style 2 --target both --doclevel --simple --model 405b --nebius --curated
# python -u generate_graph.py --style 3 --target both --doclevel --simple --model 405b --nebius --curated
# python -u generate_graph.py --style 4 --target ppi --doclevel --simple --model 405b --nebius --curated
# python -u generate_graph.py --style 4 --target tf --doclevel --simple --model 405b --nebius --curated
# python -u generate_graph.py --style 5 --target ppi --doclevel --simple --model 405b --nebius --curated
# python -u generate_graph.py --style 5 --target tf --doclevel --simple --model 405b --nebius --curated
# python -u generate_graph.py --style 6 --target ppi --doclevel --simple --model 405b --nebius --curated

# python -u graphdocs2json.py --style 1 --target tf --doclevel --simple --model 405b --nebius --curated
# python -u graphdocs2json.py --style 2 --target both --doclevel --simple --model 405b --nebius --curated
# python -u graphdocs2json.py --style 3 --target both --doclevel --simple --model 405b --nebius --curated
# python -u graphdocs2json.py --style 4 --target ppi --doclevel --simple --model 405b --nebius --curated
# python -u graphdocs2json.py --style 4 --target tf --doclevel --simple --model 405b --nebius --curated
# python -u graphdocs2json.py --style 5 --target ppi --doclevel --simple --model 405b --nebius --curated
# python -u graphdocs2json.py --style 5 --target tf --doclevel --simple --model 405b --nebius --curated
# python -u graphdocs2json.py --style 6 --target ppi --doclevel --simple --model 405b --nebius --curated
