#!/bin/bash

#SBATCH --job-name=parse_tfs
#SBATCH --output=../../outputs/slurm/parse_tfs.txt
#SBATCH --partition=long
#SBATCH --mem=50G

cd ../ &&
. ~/.venvs/test_linda/bin/activate

python -u parse_pdfs.py --converter pymupdf4llm --input-dir "/beegfs/prj/LINDA_LLM/CardioPrior/GRN_Papers" --output-dir "/prj/LINDA_LLM/outputs/parsed_papers/CardioPrior/tf"
