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
# The merged model is saved by finetune.py as:
#   {OUTPUT_ROOT}/finetunedmodels/{hf_model_id}_regulatome_{target}_merged_16bit
HF_MODEL="Meta-Llama-3.1-8B-Instruct-bnb-4bit"
MODEL_DIR="${OUTPUT_ROOT}/finetunedmodels/${HF_MODEL}_regulatome_ppi_merged_16bit"
MODEL_NAME="llama3.1:8b-regulatome-ppi"

if [[ ! -d "$MODEL_DIR" ]]; then
  echo "ERROR: Merged model not found at $MODEL_DIR"
  echo "Has the finetuning job completed?"
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

# ── Import model ───────────────────────────────────────────────────────
echo "Creating Modelfile..."
MODELFILE="/tmp/ollama_imports/Modelfile_${SLURM_JOB_ID:-local}"
mkdir -p "$(dirname "$MODELFILE")"

cat > "$MODELFILE" <<MODELFILE_EOF
FROM $MODEL_DIR

PARAMETER num_ctx 20000
PARAMETER temperature 0
PARAMETER top_p 0.9
PARAMETER seed 42

SYSTEM """You are an expert molecular biologist specializing in protein-protein interactions. Your TASK is to extract protein-protein interactions from scientific texts with high precision. You understand the difference between direct physical interactions, functional relationships, and regulatory effects. When extracting relationships, focus on evidence-based direct interactions rather than indirect associations."""
MODELFILE_EOF

echo "Importing model (this will take a few minutes)..."
ollama create "$MODEL_NAME" -f "$MODELFILE"
echo "Model '$MODEL_NAME' created successfully."

# ── Smoke test ────────────────────────────────────────────────────────
echo ""
echo "=== Smoke test ==="
ollama run "$MODEL_NAME" "Say 'PPI extraction model ready.'" 2>&1 || true

echo ""
echo "=== Done ==="
echo "Model: $MODEL_NAME"
echo "Use with: OLLAMA_HOST=127.0.0.1:$OLLAMA_PORT ollama run $MODEL_NAME"
