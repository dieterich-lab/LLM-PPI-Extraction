#!/bin/bash

#SBATCH --job-name=ll31_regu
#SBATCH --output=../../outputs/slurm/regulatome_llama31_regu.txt
#SBATCH --partition=medium
#SBATCH --mem=10G

cd ../ &&
. ~/.venvs/test_linda/bin/activate

python -u extract.py --model llama31 --node g3 --port 33  --noconfidence --loglevel off --examples negpos
python -u extract.py --model llama31regu  --node g3 --port 33  --noconfidence --loglevel off --examples negpos

