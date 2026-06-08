#!/bin/bash

#SBATCH --job-name=cardio_ppi_cp10_pos
#SBATCH --output=/beegfs/prj/LINDA_LLM/outputs/slurm/cardio_ppi_cp10_pos_%j.log
#SBATCH --partition=gpu
#SBATCH --gres=gpu:hopper:1
#SBATCH --nodelist=gpu-g5-1
#SBATCH --mem=50G

set -euo pipefail

# Ensure historical Ollama version for reproducibility.
module unload ollama || true
module load ollama/0.11.8

WORKDIR="${SLURM_SUBMIT_DIR:-}"
if [[ -z "$WORKDIR" || ! -f "$WORKDIR/scripts/extract.py" ]]; then
  if [[ -f "/prj/LINDA_LLM/scripts/extract.py" ]]; then
    WORKDIR="/prj/LINDA_LLM"
  elif [[ -f "/beegfs/prj/LINDA_LLM/scripts/extract.py" ]]; then
    WORKDIR="/beegfs/prj/LINDA_LLM"
  else
    echo "ERROR: Could not locate project root containing scripts/extract.py"
    exit 1
  fi
fi

cd "$WORKDIR/scripts"
. ~/.venvs/test_linda/bin/activate

MODEL="${MODEL:-llama33}"
LOGLEVEL="${LOGLEVEL:-info}"
FORCE_NEW="${FORCE_NEW:-1}"
EXAMPLES="${EXAMPLES:-pos}"
OLLAMA_PORT="${OLLAMA_PORT:-11439}"
PORT_SUFFIX="${PORT_SUFFIX:-39}"

DEDICATED_INPUT_DIR="/prj/LINDA_LLM/outputs/parsed_papers/CardioPrior/ppi/10paper_dedicated_20260602"
if [[ ! -d "$DEDICATED_INPUT_DIR" ]]; then
  echo "ERROR: Dedicated input directory does not exist: $DEDICATED_INPUT_DIR"
  exit 1
fi

if ! command -v ollama >/dev/null 2>&1; then
  echo "ERROR: ollama command not found on this node"
  exit 1
fi

SLURM_LOG_DIR="/beegfs/prj/LINDA_LLM/outputs/slurm"
OLLAMA_LOG="${SLURM_LOG_DIR}/ollama_cardio_ppi_cp10_pos_${SLURM_JOB_ID:-local}.log"
mkdir -p "$SLURM_LOG_DIR"

export OLLAMA_HOST="127.0.0.1:${OLLAMA_PORT}"
export OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:-1h}"
export OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-1}"
export OLLAMA_CONTEXT_LENGTH="${OLLAMA_CONTEXT_LENGTH:-80000}"
export OLLAMA_DEBUG=1

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

echo "Starting dedicated CardioPrior 10-paper PPI extraction with examples=$EXAMPLES"
echo "Model: $MODEL"
echo "Input directory override: $DEDICATED_INPUT_DIR"
echo "Ollama port: $OLLAMA_PORT"
echo "Ollama context length: $OLLAMA_CONTEXT_LENGTH"

EXTRA_ARGS=()
if [[ "$FORCE_NEW" == "1" ]]; then
  EXTRA_ARGS+=(--force_new)
fi

export LINDA_LLM_PAPER_PATH_OVERRIDE="$DEDICATED_INPUT_DIR"

python -u extract.py \
  --target ppi \
  --data cardio \
  --model "$MODEL" \
  --node local \
  --port "$PORT_SUFFIX" \
  --loglevel "$LOGLEVEL" \
  --examples "$EXAMPLES" \
  --num-shards 1 \
  --shard-index 0 \
  --ext "cardioprior10_pos_20260602" \
  "${EXTRA_ARGS[@]}"
