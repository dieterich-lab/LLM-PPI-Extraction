#!/bin/bash

#SBATCH --job-name=ll31_recall
#SBATCH --output=../../outputs/slurm/regulatome_llama31_recall.txt
#SBATCH --partition=medium
#SBATCH --mem=10G

cd ../ &&
. ~/.venvs/test_linda/bin/activate

python -u extract.py --model llama31 --node g3 --port 33  --loglevel off --recall
python -u extract.py --model llama31regu  --node g3 --port 33  --noconfidence --loglevel off --recall
python -u extract.py --model llama31 --node g3 --port 33  --loglevel off --recall --doclevel chunks
python -u extract.py --model llama31regu  --node g3 --port 33  --noconfidence --loglevel off --recall  --doclevel chunks

