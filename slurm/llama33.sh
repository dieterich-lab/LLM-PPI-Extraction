#!/bin/bash

#SBATCH --job-name=33
#SBATCH --output=/prj/LINDA_LLM/outputs/slurm/llama33.txt
#SBATCH --partition=medium
#SBATCH --mem=10G

. ~/.venvs/test_linda/bin/activate

# python -u extract.py --model llama33 --node g5 --loglevel info  --force_new &
# sleep 5
# python -u extract.py --model llama33 --node g5 --examples negpos  --force_new &
# sleep 5
# python -u extract.py --model llama33 --node g5 --dynex_k 5  --force_new &
# sleep 5
# python -u extract.py --model llama33 --node g5 --lookup  --force_new
# sleep 5
# python -u extract.py --model llama33 --node g5 --examples neg  --force_new &
# sleep 5
# python -u extract.py --model llama33 --node g5 --examples pos  --force_new &
# sleep 5
python -u extract.py --model llama33 --node g5 --ensemble  --force_new &
sleep 5
python -u extract.py --model llama33 --node g5 --tot  --force_new &
sleep 5

wait

# python -u extract.py --model llama33 --node g5 --target tf --loglevel info  --force_new &
# sleep 5
# python -u extract.py --model llama33 --node g5 --target tf --examples negpos  --force_new  &
# sleep 5
# python -u extract.py --model llama33 --node g5 --target tf  --dynex_k 5 --force_new &
# sleep 5
# python -u extract.py --model llama33 --node g5 --target tf --lookup  --force_new
# sleep 5
# python -u extract.py --model llama33 --node g5 --target tf --examples neg  --force_new  &
# sleep 5
# python -u extract.py --model llama33 --node g5 --target tf --examples pos  --force_new  &
# sleep 5
python -u extract.py --model llama33 --node g5 --target tf --ensemble  --force_new  &
sleep 5
python -u extract.py --model llama33 --node g5 --target tf --tot  --force_new  &
sleep 5

wait
