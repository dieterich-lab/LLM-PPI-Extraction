#!/bin/bash

#SBATCH --job-name=filter_cardiac_md
#SBATCH --output=/beegfs/prj/LINDA_LLM/outputs/slurm/filter_cardiac_manuscripts_%A_%a.log
#SBATCH --partition=gpu
#SBATCH --gres=gpu:turing:1
#SBATCH --nodelist=gpu-g2-1
#SBATCH --array=0-3
#SBATCH --mem=12G

set -euo pipefail

cd ../
. ~/.venvs/test_linda/bin/activate

MODEL="${MODEL:-llama31}"
MAX_CHARS="${MAX_CHARS:-60000}"
BASE_OLLAMA_PORT="${BASE_OLLAMA_PORT:-11434}"
LIMIT="${LIMIT:-0}"
NUM_SHARDS="${NUM_SHARDS:-4}"
ENABLE_COLLATE="${ENABLE_COLLATE:-1}"
SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
PORT_SUFFIX_BASE="${PORT_SUFFIX_BASE:-34}"
PORT_SUFFIX="$((PORT_SUFFIX_BASE + SHARD_INDEX))"
OLLAMA_PORT="$((BASE_OLLAMA_PORT + SHARD_INDEX))"

INPUT_DIR="/beegfs/prj/LINDA_LLM/Cardiac_Manuscripts"
OUTPUT_DIR="/beegfs/prj/LINDA_LLM/outputs/parsed_papers/Cardiac_Manuscripts/filter_cache"
RESULTS_JSONL_BASE="${RESULTS_JSONL:-/beegfs/prj/LINDA_LLM/outputs/cardiac_filter/cardiac_manuscripts_results_${MODEL}.jsonl}"
if [[ "$RESULTS_JSONL_BASE" == *.jsonl ]]; then
  RESULTS_JSONL="${RESULTS_JSONL_BASE%.jsonl}_shard${SHARD_INDEX}.jsonl"
else
  RESULTS_JSONL="${RESULTS_JSONL_BASE}_shard${SHARD_INDEX}.jsonl"
fi
SLURM_LOG_DIR="/beegfs/prj/LINDA_LLM/outputs/slurm"
OLLAMA_LOG="${SLURM_LOG_DIR}/ollama_filter_cardiac_manuscripts_${SLURM_JOB_ID:-local}_${SHARD_INDEX}.log"

mkdir -p "$SLURM_LOG_DIR"
mkdir -p "$(dirname "$RESULTS_JSONL")"
mkdir -p "$OUTPUT_DIR"

if ! command -v ollama >/dev/null 2>&1; then
  echo "ERROR: ollama command not found on this node"
  exit 1
fi

export OLLAMA_HOST="127.0.0.1:${OLLAMA_PORT}"
export OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:-1h}"
export OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-1}"
export OLLAMA_CONTEXT_LENGTH="${OLLAMA_CONTEXT_LENGTH:-32768}"

ollama serve > "$OLLAMA_LOG" 2>&1 &
OLLAMA_PID=$!

cleanup() {
  if [[ -n "${OLLAMA_PID:-}" ]] && kill -0 "$OLLAMA_PID" >/dev/null 2>&1; then
    echo "Stopping ollama server (pid=$OLLAMA_PID)"
    kill "$OLLAMA_PID" >/dev/null 2>&1 || true
    wait "$OLLAMA_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

READY=0
for _ in {1..60}; do
  if curl --silent --fail --output /dev/null "http://${OLLAMA_HOST}/api/tags"; then
    READY=1
    break
  fi
  sleep 1
done

if [[ "$READY" -ne 1 ]]; then
  echo "ERROR: ollama server did not become ready"
  echo "See: $OLLAMA_LOG"
  exit 1
fi

if [[ "${SHARD_INDEX}" == "0" ]] && [[ "$ENABLE_COLLATE" == "1" ]] && [[ -n "${SLURM_ARRAY_JOB_ID:-}" ]]; then
  COLLATE_SCRIPT="/prj/LINDA_LLM/scripts/slurm/collate_cardiac_manuscripts_shards.sh"
  if [[ -f "$COLLATE_SCRIPT" ]]; then
    COLLATE_JOB_ID=$(sbatch --parsable \
      --dependency="afterok:${SLURM_ARRAY_JOB_ID}" \
      --export=ALL,RESULTS_JSONL_BASE="$RESULTS_JSONL_BASE",NUM_SHARDS="$NUM_SHARDS" \
      "$COLLATE_SCRIPT")
    echo "Scheduled collation job ${COLLATE_JOB_ID} (afterok:${SLURM_ARRAY_JOB_ID})"
  else
    echo "WARNING: collation script not found at ${COLLATE_SCRIPT}; skipping auto-collation"
  fi
fi

echo "Starting cardiac manuscript classification"
echo "Model: $MODEL"
echo "Shard: $SHARD_INDEX/$NUM_SHARDS"
echo "Limit: $LIMIT"
echo "Ollama port: $OLLAMA_PORT"
echo "Input dir: $INPUT_DIR"
echo "Results: $RESULTS_JSONL"

poetry run python -m cardiac_filtering.filter_cardiac_papers \
  --input-dir "$INPUT_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --output-jsonl "$RESULTS_JSONL" \
  --skip-existing \
  --model "$MODEL" \
  --node local \
  --port "$PORT_SUFFIX" \
  --loglevel off \
  --max-chars "$MAX_CHARS" \
  --limit "$LIMIT" \
  --num-shards "$NUM_SHARDS" \
  --shard-index "$SHARD_INDEX"
