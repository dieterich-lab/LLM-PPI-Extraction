#!/bin/bash

#SBATCH --job-name=31_spacy
#SBATCH --output=/prj/LINDA_LLM/outputs/slurm/llama31_spacy.txt
#SBATCH --partition=medium
#SBATCH --mem=10G

. ~/.venvs/test_linda/bin/activate

# python -u extract.py --model llama31 --loglevel info --spacy_nes_given --force_new &
# sleep 5
# python -u extract.py --model llama31 --examples negpos --spacy_nes_given --force_new &
# sleep 5
# python -u extract.py --model llama31 --dynex_k 5 --spacy_nes_given --force_new &
# sleep 5
# python -u extract.py --model llama31 --lookup --spacy_nes_given --force_new
# sleep 5

# wait

# python -u extract.py --model llama31 --target tf --loglevel info --spacy_nes_given --force_new &
# sleep 5
# python -u extract.py --model llama31 --target tf --examples negpos --spacy_nes_given --force_new  &
# sleep 5
# python -u extract.py --model llama31 --target tf --spacy_nes_given --dynex_k 5 --force_new &
# sleep 5
# python -u extract.py --model llama31 --target tf --lookup --spacy_nes_given --force_new
# sleep 5

# wait

# python -u extract.py --model llama31 --examples neg --spacy_nes_given --force_new &
# sleep 5
# python -u extract.py --model llama31 --examples pos --spacy_nes_given --force_new &
# sleep 5
python -u extract.py --model llama31 --ensemble --spacy_nes_given --force_new &
sleep 5
python -u extract.py --model llama31 --tot --spacy_nes_given --force_new &
sleep 5

# wait

python -u extract.py --model llama31 --target tf --ensemble --spacy_nes_given --force_new  &
sleep 5
python -u extract.py --model llama31 --target tf --tot --spacy_nes_given --force_new  &
sleep 5

wait