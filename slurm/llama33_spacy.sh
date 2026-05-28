#!/bin/bash

#SBATCH --job-name=33_spacy
#SBATCH --output=/prj/LINDA_LLM/outputs/slurm/llama33_spacy.txt
#SBATCH --partition=medium
#SBATCH --mem=10G

. ~/.venvs/test_linda/bin/activate

# python -u extract.py --model llama33 --node g5 --loglevel info --spacy_nes_given --force_new &
# sleep 5
# python -u extract.py --model llama33 --node g5 --examples negpos --spacy_nes_given --force_new &
# sleep 5
# python -u extract.py --model llama33 --node g5 --dynex_k 5 --spacy_nes_given --force_new &
# sleep 5
# python -u extract.py --model llama33 --node g5 --lookup --spacy_nes_given --force_new
# sleep 5

# wait

# python -u extract.py --model llama33 --node g5 --target tf --loglevel info --spacy_nes_given --force_new &
# sleep 5
# python -u extract.py --model llama33 --node g5 --target tf --examples negpos --spacy_nes_given --force_new  &
# sleep 5
# python -u extract.py --model llama33 --node g5 --target tf --spacy_nes_given --dynex_k 5 --force_new &
# sleep 5
# python -u extract.py --model llama33 --node g5 --target tf --lookup --spacy_nes_given --force_new
# sleep 5

# wait

# python -u extract.py --model llama33 --node g5 --examples neg --spacy_nes_given --force_new &
# sleep 5
# python -u extract.py --model llama33 --node g5 --examples pos --spacy_nes_given --force_new &
# sleep 5
python -u extract.py --model llama33 --node g5 --ensemble --spacy_nes_given --force_new &
sleep 5
python -u extract.py --model llama33 --node g5 --tot --spacy_nes_given --force_new &
sleep 5

wait

# python -u extract.py --model llama33 --node g5 --target tf --examples neg --spacy_nes_given --force_new  &
# sleep 5
# python -u extract.py --model llama33 --node g5 --target tf --examples pos --spacy_nes_given --force_new  &
# sleep 5
python -u extract.py --model llama33 --node g5 --target tf --ensemble --spacy_nes_given --force_new  &
sleep 5
python -u extract.py --model llama33 --node g5 --target tf --tot --spacy_nes_given --force_new  &
sleep 5

wait