#!/bin/bash

#SBATCH --job-name=bench_ppi_ctx
#SBATCH --output=/beegfs/prj/LINDA_LLM/outputs/slurm/bench_ppi_ctx_%j.log
#SBATCH --partition=gpu
#SBATCH --gres=gpu:hopper:1
#SBATCH --nodelist=gpu-g5-1
#SBATCH --mem=24G

set -euo pipefail

cd /prj/LINDA_LLM/scripts
. ~/.venvs/test_linda/bin/activate

MODEL="${MODEL:-llama33}"
MAX_CHARS="${MAX_CHARS:-20000}"
N_DOCS="${N_DOCS:-3}"
CHUNKSIZE="${CHUNKSIZE:-2001}"
BASE_OLLAMA_PORT="${BASE_OLLAMA_PORT:-11434}"
PORT_SUFFIX="${PORT_SUFFIX:-34}"
RUN_SLOT="${RUN_SLOT:-0}"
OLLAMA_PORT="$BASE_OLLAMA_PORT"
LOGLEVEL="${LOGLEVEL:-off}"
DATASET="${DATASET:-cardiac}"
TARGET="${TARGET:-ppi}"
PAPER_DIR_OVERRIDE="${PAPER_DIR_OVERRIDE:-/beegfs/prj/LINDA_LLM/Cardiac_Manuscripts}"

if (( RUN_SLOT < 0 || RUN_SLOT > 3 )); then
  echo "ERROR: RUN_SLOT must be between 0 and 3"
  exit 1
fi

OLLAMA_PORT="$((BASE_OLLAMA_PORT + RUN_SLOT))"
PORT_SUFFIX="$((PORT_SUFFIX + RUN_SLOT))"

SLURM_LOG_DIR="/beegfs/prj/LINDA_LLM/outputs/slurm"
mkdir -p "$SLURM_LOG_DIR"
OLLAMA_LOG="${SLURM_LOG_DIR}/ollama_bench_ppi_ctx_${SLURM_JOB_ID:-local}.log"

if ! command -v ollama >/dev/null 2>&1; then
  echo "ERROR: ollama command not found"
  exit 1
fi

export OLLAMA_HOST="127.0.0.1:${OLLAMA_PORT}"
export OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:-1h}"
export OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-1}"
export OLLAMA_CONTEXT_LENGTH="${OLLAMA_CONTEXT_LENGTH:-128000}"
export MAX_CHARS="$MAX_CHARS"
export LINDA_LLM_PAPER_PATH_OVERRIDE="$PAPER_DIR_OVERRIDE"

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

echo "Starting quick PPI context benchmark"
echo "Model: $MODEL"
echo "MAX_CHARS: $MAX_CHARS"
echo "N_DOCS: $N_DOCS"
echo "CHUNKSIZE: $CHUNKSIZE"
echo "Dataset: $DATASET"
echo "Target: $TARGET"
echo "Paper dir override: $PAPER_DIR_OVERRIDE"
echo "Run slot: $RUN_SLOT"
echo "Ollama host: 127.0.0.1:$OLLAMA_PORT"
echo "Client port suffix: $PORT_SUFFIX"

echo "Start timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
START_TS=$(date +%s)

python -u extract.py \
  --target "$TARGET" \
  --data "$DATASET" \
  --model "$MODEL" \
  --node local \
  --port "$PORT_SUFFIX" \
  --loglevel "$LOGLEVEL" \
  --chunksize "$CHUNKSIZE" \
  --startfromdoc 0 \
  --untildoc "$N_DOCS" \
  --force_new \
  --ext "bench_${MODEL}_${MAX_CHARS}_${SLURM_JOB_ID:-local}"

END_TS=$(date +%s)
DURATION=$((END_TS - START_TS))
echo "End timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Total duration seconds: $DURATION"
