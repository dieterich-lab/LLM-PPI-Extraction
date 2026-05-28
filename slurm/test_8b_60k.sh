#!/bin/bash

#SBATCH --job-name=test_8b_60k
#SBATCH --output=/beegfs/prj/LINDA_LLM/outputs/slurm/test_8b_60k_%j.log
#SBATCH --partition=gpu
#SBATCH --gres=gpu:hopper:1
#SBATCH --nodelist=gpu-g5-1
#SBATCH --mem=20G

set -euo pipefail

cd /prj/LINDA_LLM/scripts
. ~/.venvs/test_linda/bin/activate

export MAX_CHARS=60000
export OLLAMA_HOST="127.0.0.1:11434"
export OLLAMA_KEEP_ALIVE="1h"
export OLLAMA_NUM_PARALLEL="1"
export OLLAMA_CONTEXT_LENGTH="128000"
export OLLAMA_DEBUG=1

OLLAMA_LOG="/beegfs/prj/LINDA_LLM/outputs/slurm/ollama_test_8b_60k_${SLURM_JOB_ID}.log"
mkdir -p "$(dirname "$OLLAMA_LOG")"

# Start ollama
ollama serve > "$OLLAMA_LOG" 2>&1 &
OLLAMA_PID=$!

cleanup() {
  if [[ -n "${OLLAMA_PID:-}" ]] && kill -0 "$OLLAMA_PID" 2>/dev/null; then
    echo "Stopping ollama"
    kill "$OLLAMA_PID" 2>/dev/null || true
    wait "$OLLAMA_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

# Wait for ollama
READY=0
for _ in {1..60}; do
  if curl --silent --fail "http://127.0.0.1:11434/api/tags" >/dev/null 2>&1; then
    READY=1
    break
  fi
  sleep 1
done

if [[ "$READY" -ne 1 ]]; then
  echo "ERROR: ollama did not become ready"
  exit 1
fi

echo "====================================================================="
echo "Testing llama3.1:8b with MAX_CHARS=60000 (3 documents)"
echo "====================================================================="

START_TIME=$(date +%s)

# Run extraction with llama31 model
python3 extract.py --model llama31 --node local --data cardio --target ppi --doclevel docs

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
echo "====================================================================="
echo "Test complete"
echo "Duration: ${DURATION} seconds"
echo "Check ollama log at: $OLLAMA_LOG"
echo "====================================================================="
