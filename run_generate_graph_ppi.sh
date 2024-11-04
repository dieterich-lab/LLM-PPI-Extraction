#!/bin/bash

#SBATCH --job-name=generate_graph_ppi
#SBATCH --output=../outputs/slurm/generate_graph_ppi.txt
#SBATCH --partition=long
#SBATCH --mem=100G

python -u generate_graph.py --model biollm  --parser llama_parse --curated --task ppi
python -u generate_graph.py --model biollm --simple --parser llama_parse --curated --task ppi
python -u generate_graph.py --model biollm  --parser marker --curated --task ppi
python -u generate_graph.py --model biollm --simple --parser marker --curated --task ppi

# python -u generate_graph.py --model biollm  --parser llama_parse
# python -u generate_graph.py --model biollm --simple --parser llama_parse
# python -u generate_graph.py --model 70b  --parser llama_parse
# python -u generate_graph.py --model 70b --simple --parser llama_parse
# python -u generate_graph.py --model biollm  --parser marker
# python -u generate_graph.py --model biollm --simple --parser marker
# python -u generate_graph.py --model 70b  --parser marker
# python -u generate_graph.py --model 70b --simple --parser marker
