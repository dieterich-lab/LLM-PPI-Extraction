#!/bin/bash

#SBATCH --job-name=ll31_true
#SBATCH --output=../../outputs/slurm/llama33_true.txt
#SBATCH --partition=medium
#SBATCH --mem=10G

cd ../ &&
. ~/.venvs/test_linda/bin/activate

python -u extract.py --target ppi --model llama33regu  --node g4
python -u extract.py --target ppi --model llama33regu  --node g4 --examples pos
python -u extract.py --target ppi --model llama33regu  --node g4 --examples neg
python -u extract.py --target ppi --model llama33regu  --node g4 --examples negpos

python -u extract.py --target ppi --model llama33  --true_ners_given --node g4 
python -u extract.py --target ppi --model llama33  --examples pos --true_ners_given --node g4 
python -u extract.py --target ppi --model llama33  --examples neg --true_ners_given --node g4 
python -u extract.py --target ppi --model llama33  --examples negpos --true_ners_given --node g4 

python -u extract.py --target tf --model llama33  --true_ners_given --node g4 
python -u extract.py --target tf --model llama33  --examples pos --true_ners_given --node g4 
python -u extract.py --target tf --model llama33  --examples neg --true_ners_given --node g4 
python -u extract.py --target tf --model llama33  --examples negpos --true_ners_given --node g4 

python -u extract.py --target ppi --model llama33  --all_ners_given --node g4
python -u extract.py --target ppi --model llama33  --examples pos --all_ners_given --node g4
python -u extract.py --target ppi --model llama33  --examples neg --all_ners_given --node g4
python -u extract.py --target ppi --model llama33  --examples negpos --all_ners_given --node g4

python -u extract.py --target tf --model llama33  --all_ners_given --node g4
python -u extract.py --target tf --model llama33  --examples pos --all_ners_given --node g4
python -u extract.py --target tf --model llama33  --examples neg --all_ners_given --node g4
python -u extract.py --target tf --model llama33  --examples negpos --all_ners_given --node g4

python -u extract.py --target tf --model llama33regu  --all_ners_given --node g4
python -u extract.py --target tf --model llama33regu  --examples pos --all_ners_given --node g4
python -u extract.py --target tf --model llama33regu  --examples neg --all_ners_given --node g4
python -u extract.py --target tf --model llama33regu  --examples negpos --all_ners_given --node g4

python -u extract.py --target ppi --model llama33regu  --all_ners_given --node g4
python -u extract.py --target ppi --model llama33regu  --examples pos --all_ners_given --node g4
python -u extract.py --target ppi --model llama33regu  --examples neg --all_ners_given --node g4
python -u extract.py --target ppi --model llama33regu  --examples negpos --all_ners_given --node g4

python -u extract.py --target tf --model llama33regu  --true_ners_given --node g4 
python -u extract.py --target tf --model llama33regu  --examples pos --true_ners_given --node g4 
python -u extract.py --target tf --model llama33regu  --examples neg --true_ners_given --node g4 
python -u extract.py --target tf --model llama33regu  --examples negpos --true_ners_given --node g4 

python -u extract.py --target ppi --model llama33regu  --true_ners_given --node g4 
python -u extract.py --target ppi --model llama33regu  --examples pos --true_ners_given --node g4 
python -u extract.py --target ppi --model llama33regu  --examples neg --true_ners_given --node g4 
python -u extract.py --target ppi --model llama33regu  --examples negpos --true_ners_given --node g4 
