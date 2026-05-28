#!/bin/bash

#SBATCH --job-name=31
#SBATCH --output=/prj/LINDA_LLM/outputs/slurm/llama31.txt
#SBATCH --partition=medium
#SBATCH --mem=10G

. ~/.venvs/test_linda/bin/activate

# python -u extract.py --model llama31 --loglevel info  --force_new &
# sleep 5
# python -u extract.py --model llama31 --examples negpos  --force_new &
# sleep 5
# python -u extract.py --model llama31 --dynex_k 5  --force_new &
# sleep 5
# python -u extract.py --model llama31 --lookup  --force_new
# sleep 5
# python -u extract.py --model llama31 --examples neg  --force_new &
# sleep 5
# python -u extract.py --model llama31 --examples pos  --force_new &
# sleep 5
python -u extract.py --model llama31 --ensemble  --force_new &
sleep 5
python -u extract.py --model llama31 --tot  --force_new &
sleep 5


wait

# python -u extract.py --model llama31 --target tf --loglevel info  --force_new &
# sleep 5
# python -u extract.py --model llama31 --target tf --examples negpos  --force_new  &
# sleep 5
# python -u extract.py --model llama31 --target tf  --dynex_k 5 --force_new &
# sleep 5
# python -u extract.py --model llama31 --target tf --lookup  --force_new
# sleep 5
# python -u extract.py --model llama31 --target tf --examples neg  --force_new  &
# sleep 5
# python -u extract.py --model llama31 --target tf --examples pos  --force_new  &
# sleep 5
python -u extract.py --model llama31 --target tf --ensemble  --force_new  &
sleep 5
python -u extract.py --model llama31 --target tf --tot  --force_new  &
sleep 5

wait
