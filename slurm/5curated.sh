#!/bin/bash

#SBATCH --job-name=5curated
#SBATCH --output=../../outputs/slurm/5curated.txt
#SBATCH --partition=medium
#SBATCH --mem=10G

cd ../ &&
. ~/.venvs/test_linda/bin/activate

# python -u extract.py --model llama31 --node g5 --port 35 --loglevel off --data 5curated --doclevel chunks
# python -u extract.py --model llama31 --node g5 --port 35 --loglevel off --data 5curated  --examples neg --doclevel chunks
# python -u extract.py --model llama31 --node g5 --port 35 --loglevel off --data 5curated  --examples pos --doclevel chunks
# python -u extract.py --model llama31 --node g5 --port 35 --loglevel off --data 5curated  --examples negpos --doclevel chunks
# python -u extract.py --model llama31regu --node g5 --port 35 --loglevel off --data 5curated --doclevel chunks
# python -u extract.py --model llama31regu --node g5 --port 35 --loglevel off --data 5curated  --examples neg --doclevel chunks
# python -u extract.py --model llama31regu --node g5 --port 35 --loglevel off --data 5curated  --examples pos --doclevel chunks
# python -u extract.py --model llama31regu --node g5 --port 35 --loglevel off --data 5curated  --examples negpos --doclevel chunks
# python -u extract.py --model llama33 --node g4 --port 34 --loglevel off --data 5curated --doclevel chunks
# python -u extract.py --model llama33 --node g4 --port 34 --loglevel off --data 5curated  --examples neg --doclevel chunks
# python -u extract.py --model llama33 --node g4 --port 34 --loglevel off --data 5curated  --examples pos --doclevel chunks
# python -u extract.py --model llama33 --node g4 --port 34 --loglevel off --data 5curated  --examples negpos --doclevel chunks
# python -u extract.py --model llama33regu --node g4 --port 34 --loglevel off --data 5curated --doclevel chunks
# python -u extract.py --model llama33regu --node g4 --port 34 --loglevel off --data 5curated  --examples neg --doclevel chunks
# python -u extract.py --model llama33regu --node g4 --port 34 --loglevel off --data 5curated  --examples pos --doclevel chunks
# python -u extract.py --model llama33regu --node g4 --port 34 --loglevel off --data 5curated  --examples negpos --doclevel chunks


# python -u extract.py --model llama31 --node g4 --port 34 --loglevel off --data 5curated  --chattype lookup --extractionmode nerrel --doclevel chunks
# python -u extract.py --model llama31regu --node g4 --port 34 --loglevel off --data 5curated  --chattype lookup --extractionmode nerrel --doclevel chunks
# python -u extract.py --model llama33 --node g4 --port 34 --loglevel off --data 5curated  --chattype lookup --extractionmode nerrel --doclevel chunks
# python -u extract.py --model llama33regu --node g4 --port 34 --loglevel off --data 5curated  --chattype lookup --extractionmode nerrel --doclevel chunks

# python -u extract.py --model llama31 --node g4 --port 34 --loglevel off --data 5curated  --dynex --doclevel chunks
# python -u extract.py --model llama31regu --node g4 --port 34 --loglevel off --data 5curated  --dynex --doclevel chunks
# python -u extract.py --model llama33 --node g4 --port 34 --loglevel off --data 5curated  --dynex --doclevel chunks
# python -u extract.py --model llama33regu --node g4 --port 34 --loglevel off --data 5curated  --dynex --doclevel chunks

# python -u extract.py --model llama31 --node g5 --port 35 --loglevel off --data 5curated --doclevel chunks
# python -u extract.py --model llama31regu --node g5 --port 35 --loglevel off --data 5curated --doclevel chunks 
# python -u extract.py --model llama31 --node g5 --port 35 --loglevel off --data 5curated --doclevel chunks --chunksize 4000
# python -u extract.py --model llama31regu --node g5 --port 35 --loglevel off --data 5curated --doclevel chunks  --chunksize 4000
# python -u extract.py --model llama31 --node g5 --port 35 --loglevel off --data 5curated --doclevel chunks --chunksize 8000
# python -u extract.py --model llama31regu --node g5 --port 35 --loglevel off --data 5curated --doclevel chunks  --chunksize 8000
# python -u extract.py --model llama31 --node g5 --port 35 --loglevel off --data 5curated --doclevel chunks --chunksize 16000
# python -u extract.py --model llama31regu --node g5 --port 35 --loglevel off --data 5curated --doclevel chunks  --chunksize 16000

python -u extract.py --model llama33 --node g4 --port 34 --loglevel off --data 5curated --doclevel chunks --chunksize 8000
python -u extract.py --model llama33regu --node g4 --port 34 --loglevel off --data 5curated --doclevel chunks  --chunksize 8000
python -u extract.py --model llama33 --node g4 --port 34 --loglevel off --data 5curated --doclevel chunks --chunksize 16000
python -u extract.py --model llama33regu --node g4 --port 34 --loglevel off --data 5curated --doclevel chunks  --chunksize 16000
