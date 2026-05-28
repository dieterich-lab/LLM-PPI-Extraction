#!/bin/bash

#SBATCH --job-name=parse_ppis
#SBATCH --output=../../outputs/slurm/parse_ppis.txt
#SBATCH --partition=medium
#SBATCH --mem=50G

cd ../ &&
. ~/.venvs/test_linda/bin/activate

python -u parse_pdfs.py --converter pymupdf4llm --input-dir "/beegfs/prj/LINDA_LLM/CardioPrior/PPI_Papers/" --output-dir "/prj/LINDA_LLM/outputs/parsed_papers/CardioPrior/ppi"
