#!/bin/bash
#
# Import the fine-tuned Llama 3.1 8B (RegulaTome PPI) model into Ollama.
#
# Reads the merged 16-bit safetensors model saved by finetune.py and
# registers it as 'llama3.1:8b-regulatome-ppi' in Ollama.
#
# Prerequisites:
#   - Finetuning must have completed (model saved to .../ppi_merged_16bit)
#   - Ollama must be installed on the node
#
# Usage:
#   sbatch slurm/ollama_import_llama31_regu_ppi.sh

#SBATCH --job-name=ollama_import
#SBATCH --output=../outputs/slurm/ollama_import_llama31_regu_ppi_%j.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:ampere:1
#SBATCH --nodelist=gpu-g4-1
#SBATCH --mem=50G

set -euo pipefail

# ── Load .env configuration ────────────────────────────────────────────
if [[ -f "${SLURM_SUBMIT_DIR:-.}/.env" ]]; then
  set -a; source "${SLURM_SUBMIT_DIR:-.}/.env"; set +a
fi

PROJECT_ROOT="${LINDA_LLM_PROJECT_ROOT:-/prj/LINDA_LLM}"
SCRIPTS_DIR="$PROJECT_ROOT/scripts"
OUTPUT_ROOT="${LINDA_LLM_OUTPUT_ROOT:-$PROJECT_ROOT/outputs}"

# ── Model paths ───────────────────────────────────────────────────────
# Finetuning saves three formats; we use the GGUF (quantized, has Modelfile).
# Path: {OUTPUT_ROOT}/finetunedmodels/unsloth/{hf_model_id}_regulatome_ppi_GGUF
MODEL_DIR="${OUTPUT_ROOT}/finetunedmodels/unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit_regulatome_ppi_GGUF"
MODEL_NAME="llama3.1:8b-regulatome-ppi"

if [[ ! -f "$MODEL_DIR/Modelfile" ]]; then
  echo "ERROR: Modelfile not found at $MODEL_DIR/Modelfile"
  echo "Has the finetuning job (--save) completed?"
  exit 1
fi

echo "=== Ollama Model Import ==="
echo "Source:  $MODEL_DIR"
echo "Target:  $MODEL_NAME"
echo ""

# ── Start Ollama server ───────────────────────────────────────────────
OLLAMA_PORT="11439"
export OLLAMA_HOST="127.0.0.1:${OLLAMA_PORT}"
export OLLAMA_KEEP_ALIVE="4h"
export OLLAMA_NUM_PARALLEL=1

LOG_DIR="${LINDA_LLM_SLURM_LOG_DIR:-$OUTPUT_ROOT/slurm}"
mkdir -p "$LOG_DIR"
OLLAMA_LOG="${LOG_DIR}/ollama_import_${SLURM_JOB_ID:-local}.log"

if ! command -v ollama >/dev/null 2>&1; then
  echo "ERROR: ollama command not found on this node"
  exit 1
fi

ollama serve > "$OLLAMA_LOG" 2>&1 &
OLLAMA_PID=$!

cleanup() {
  if [[ -n "${OLLAMA_PID:-}" ]] && kill -0 "$OLLAMA_PID" >/dev/null 2>&1; then
    kill "$OLLAMA_PID" >/dev/null 2>&1 || true
    wait "$OLLAMA_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

# ── Wait for Ollama ───────────────────────────────────────────────────
echo "Waiting for Ollama server on port $OLLAMA_PORT..."
for _ in {1..60}; do
  if curl --silent --fail --output /dev/null "http://${OLLAMA_HOST}/api/tags"; then
    break
  fi
  sleep 1
done

if ! curl --silent --fail --output /dev/null "http://${OLLAMA_HOST}/api/tags"; then
  echo "ERROR: Ollama server did not start"
  exit 1
fi
echo "Ollama ready."

# ── Import model using Unsloth-generated Modelfile ─────────────────────
echo "Importing model from $MODEL_DIR (this will take a few minutes)..."
echo "Modelfile contents:"
cat "$MODEL_DIR/Modelfile"
echo ""

# Remove existing model if present (to allow re-import)
if ollama list 2>/dev/null | grep -q "$MODEL_NAME"; then
  echo "Removing existing model '$MODEL_NAME'..."
  ollama rm "$MODEL_NAME" 2>/dev/null || true
fi

ollama create "$MODEL_NAME" -f "$MODEL_DIR/Modelfile"
echo "Model '$MODEL_NAME' created successfully."

# ── Smoke test ────────────────────────────────────────────────────────
echo ""
echo "=== Smoke test ==="
ollama run "$MODEL_NAME" "Say 'PPI extraction model ready.'" 2>&1 || true

echo ""
echo "=== Done ==="
echo "Model: $MODEL_NAME"
echo "Use with: OLLAMA_HOST=127.0.0.1:$OLLAMA_PORT ollama run $MODEL_NAME"
