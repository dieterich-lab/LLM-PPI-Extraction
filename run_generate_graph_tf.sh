#!/bin/bash

#SBATCH --job-name=generate_tf
#SBATCH --output=../outputs/slurm/generate_graph_tf.txt
#SBATCH --partition=long
#SBATCH --mem=100G

python -u generate_graph.py --model biollm  --parser llama_parse --curated --task tf
python -u generate_graph.py --model biollm --simple --parser llama_parse --curated --task tf
python -u generate_graph.py --model biollm  --parser marker --curated --task tf
python -u generate_graph.py --model biollm --simple --parser marker --curated --task tf

# python -u generate_graph.py --model biollm  --parser llama_parse
# python -u generate_graph.py --model biollm --simple --parser llama_parse
# python -u generate_graph.py --model 70b  --parser llama_parse
# python -u generate_graph.py --model 70b --simple --parser llama_parse
# python -u generate_graph.py --model biollm  --parser marker
# python -u generate_graph.py --model biollm --simple --parser marker
# python -u generate_graph.py --model 70b  --parser marker
# python -u generate_graph.py --model 70b --simple --parser marker
