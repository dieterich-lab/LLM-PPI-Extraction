#!/bin/bash

#SBATCH --job-name=graphdocs2json
#SBATCH --output=../outputs/slurm/graphdocs2json.txt
#SBATCH --partition=long
#SBATCH --mem=100G

python -u graphdocs2json.py
