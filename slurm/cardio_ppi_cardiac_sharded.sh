#!/bin/bash
#
# Sharded cardiac PPI extraction via Llama 3.3 70B (Ollama).
# Each SLURM array task runs one shard with its own Ollama instance.
#
# Prerequisites:
#   Set LINDA_LLM_PROJECT_ROOT and LINDA_LLM_PYTHON_VENV in your .env
#   or export them before submitting.
#
# Usage:
#   sbatch slurm/cardio_ppi_cardiac_sharded.sh

#SBATCH --job-name=cardio_ppi_cardiac
#SBATCH --output=${LINDA_LLM_PROJECT_ROOT:-.}/outputs/slurm/cardio_ppi_cardiac_%A_%a.log
#SBATCH --partition=gpu
#SBATCH --gres=gpu:hopper:1
#SBATCH --nodelist=gpu-g5-1
#SBATCH --array=0-1
#SBATCH --mem=50G

set -euo pipefail

# ── Load .env configuration ────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [[ -f "$SCRIPT_DIR/../.env" ]]; then
  set -a; source "$SCRIPT_DIR/../.env"; set +a
fi

# ── Project root discovery ─────────────────────────────────────────────
if [[ -n "${LINDA_LLM_PROJECT_ROOT:-}" ]]; then
  WORKDIR="$LINDA_LLM_PROJECT_ROOT"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/scripts/extract.py" ]]; then
  WORKDIR="$SLURM_SUBMIT_DIR"
else
  echo "ERROR: Set LINDA_LLM_PROJECT_ROOT or submit from the project root."
  exit 1
fi

cd "$WORKDIR/scripts"

# ── Python venv ────────────────────────────────────────────────────────
VENV="${LINDA_LLM_PYTHON_VENV:-${HOME}/.venvs/test_linda}"
if [[ -f "$VENV/bin/activate" ]]; then
  . "$VENV/bin/activate"
else
  echo "ERROR: Python venv not found at $VENV"
  exit 1
fi

MODEL="${MODEL:-llama33}"
NUM_SHARDS="${NUM_SHARDS:-2}"
SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
BASE_OLLAMA_PORT="${BASE_OLLAMA_PORT:-11434}"
PORT_SUFFIX_BASE="${PORT_SUFFIX_BASE:-34}"
PORT_SUFFIX="$((PORT_SUFFIX_BASE + SHARD_INDEX))"
OLLAMA_PORT="$((BASE_OLLAMA_PORT + SHARD_INDEX))"
LOGLEVEL="${LOGLEVEL:-info}"
FORCE_NEW="${FORCE_NEW:-0}"

if (( NUM_SHARDS < 1 || NUM_SHARDS > 4 )); then
  echo "ERROR: NUM_SHARDS must be between 1 and 4"
  exit 1
fi

# Keep array fixed at 4 tasks and no-op extra tasks when NUM_SHARDS < 4.
if (( SHARD_INDEX >= NUM_SHARDS )); then
  echo "Skipping task ${SHARD_INDEX}; NUM_SHARDS=${NUM_SHARDS}"
  exit 0
fi

SLURM_LOG_DIR="${LINDA_LLM_SLURM_LOG_DIR:-$WORKDIR/outputs/slurm}"
OLLAMA_LOG="${SLURM_LOG_DIR}/ollama_cardio_ppi_cardiac_${SLURM_JOB_ID:-local}_${SHARD_INDEX}.log"
mkdir -p "$SLURM_LOG_DIR"

if ! command -v ollama >/dev/null 2>&1; then
  echo "ERROR: ollama command not found on this node"
  exit 1
fi

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

echo "Starting cardiac PPI extraction"
echo "Model: $MODEL"
echo "Shard: $SHARD_INDEX/$NUM_SHARDS"
echo "Ollama port: $OLLAMA_PORT"
echo "Ollama context length: $OLLAMA_CONTEXT_LENGTH"

EXTRA_ARGS=()
if [[ "$FORCE_NEW" == "1" ]]; then
  EXTRA_ARGS+=(--force_new)
fi

python -u extract.py \
  --target ppi \
  --data cardiac \
  --model "$MODEL" \
  --node local \
  --port "$PORT_SUFFIX" \
  --loglevel "$LOGLEVEL" \
  --num-shards "$NUM_SHARDS" \
  --shard-index "$SHARD_INDEX" \
  --ext "cardiac_shard${SHARD_INDEX}" \
  "${EXTRA_ARGS[@]}"
