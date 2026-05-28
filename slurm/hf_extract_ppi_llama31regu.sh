#!/bin/bash
#SBATCH --job-name=hf_ppi_31regu
#SBATCH --output=/beegfs/prj/LINDA_LLM/outputs/slurm/hf_extract_ppi_%j.log
#SBATCH --partition=gpu
#SBATCH --gres=gpu:ampere:1
#SBATCH --nodelist=gpu-g4-1
#SBATCH --mem=64G

set -euo pipefail

cd /prj/LINDA_LLM/scripts
. ~/.venvs/test_linda/bin/activate

MODEL_ALIAS="${MODEL_ALIAS:-llama31regu}"
INPUT_DIR="${INPUT_DIR:-/beegfs/prj/LINDA_LLM/Cardiac_Manuscripts_test10}"
OUTPUT_JSONL="${OUTPUT_JSONL:-/beegfs/prj/LINDA_LLM/outputs/triples/cardio/ppi/llama31regu/hf_local_test/triples_hf_${SLURM_JOB_ID:-local}.jsonl}"
MAX_CHARS="${MAX_CHARS:-20000}"
START_IDX="${START_IDX:-0}"
LIMIT="${LIMIT:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"

mkdir -p /beegfs/prj/LINDA_LLM/outputs/slurm
mkdir -p "$(dirname "$OUTPUT_JSONL")"

echo "Starting HF local extraction via transformers"
echo "Model alias: $MODEL_ALIAS"
echo "Input dir: $INPUT_DIR"
echo "Output: $OUTPUT_JSONL"
echo "MAX_CHARS: $MAX_CHARS"
echo "START_IDX: $START_IDX"
echo "LIMIT: $LIMIT"
echo "MAX_NEW_TOKENS: $MAX_NEW_TOKENS"
echo "Start timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
START_TS=$(date +%s)

~/.venvs/test_linda/bin/python hf_extract_ppi.py \
  --model "$MODEL_ALIAS" \
  --input-dir "$INPUT_DIR" \
  --output-jsonl "$OUTPUT_JSONL" \
  --max-chars "$MAX_CHARS" \
  --start "$START_IDX" \
  --limit "$LIMIT" \
  --max-new-tokens "$MAX_NEW_TOKENS"

END_TS=$(date +%s)
DURATION=$((END_TS - START_TS))
echo "End timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Total duration seconds: $DURATION"
