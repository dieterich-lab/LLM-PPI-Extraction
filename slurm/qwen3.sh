#!/bin/bash

#SBATCH --job-name=q
#SBATCH --output=../../outputs/slurm/qwen3.txt
#SBATCH --partition=medium
#SBATCH --mem=10G

cd ../ &&
. ~/.venvs/test_linda/bin/activate

# python -u extract.py --target tf --model qwen3 --node g2
# python -u extract.py --target tf --model qwen3 --node g2 --examples pos
# python -u extract.py --target tf --model qwen3 --node g2 --examples neg
# python -u extract.py --target tf --model qwen3 --node g2 --examples negpos

python -u extract.py --target tf --model qwen332 
python -u extract.py --target tf --model qwen332 --examples pos
python -u extract.py --target tf --model qwen332 --examples neg
python -u extract.py --target tf --model qwen332 --examples negpos
