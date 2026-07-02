#!/bin/bash
#
# Generate protein synonyms for the regulatome PPI dataset.
# Reads the full-corpus direct_normal extraction (job 660834, 1591 docs) and
# calls CreateAltNames via BAML for every unique protein entity in the triples.
#
# Output: .../llama33/direct/oneshot/docs/synonyms_direct_normal_20260615_660834.json
#   (written next to the source .jsonl by synonyms.py)

#SBATCH --job-name=regu_ppi_synonyms
#SBATCH --output=/beegfs/prj/LINDA_LLM/outputs/slurm/regu_ppi_synonyms_%j.log
#SBATCH --partition=gpu
#SBATCH --gres=gpu:hopper:1
#SBATCH --nodelist=gpu-g5-1
#SBATCH --mem=60G

set -euo pipefail

OLLAMA_PORT="11437"
OLLAMA_LOG_DIR="/beegfs/prj/LINDA_LLM/outputs/slurm"

WORKDIR="/beegfs/prj/LINDA_LLM/scripts"
if [[ ! -f "$WORKDIR/synonyms.py" ]]; then
  echo "ERROR: scripts dir not found at $WORKDIR"
  exit 1
fi

cleanup() {
  if [[ -n "${OLLAMA_PID:-}" ]] && kill -0 "$OLLAMA_PID" >/dev/null 2>&1; then
    kill "$OLLAMA_PID" >/dev/null 2>&1 || true
    wait "$OLLAMA_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

wait_for_ollama() {
  local url="$1"
  for _ in {1..90}; do
    if curl --silent --fail --output /dev/null "$url"; then
      return 0
    fi
    sleep 1
  done
  return 1
}

. ~/.venvs/test_linda/bin/activate
cd "$WORKDIR"

export OLLAMA_HOST="127.0.0.1:${OLLAMA_PORT}"
export OLLAMA_KEEP_ALIVE="4h"
export OLLAMA_NUM_PARALLEL=1
export OLLAMA_CONTEXT_LENGTH=80000
export OLLAMA_DEBUG=1
unset MAX_CHARS

OLLAMA_LOG="${OLLAMA_LOG_DIR}/ollama_regu_ppi_synonyms_${SLURM_JOB_ID:-local}.log"
ollama serve > "$OLLAMA_LOG" 2>&1 &
OLLAMA_PID=$!

if ! wait_for_ollama "http://127.0.0.1:${OLLAMA_PORT}/api/tags"; then
  echo "ERROR: Ollama did not become ready on port ${OLLAMA_PORT}"
  exit 1
fi
echo "Ollama ready on port ${OLLAMA_PORT}"

python -u synonyms.py \
  --model llama33 \
  --node local \
  --port 37 \
  --chattype oneshot \
  --data regulatome \
  --target ppi \
  --doclevel docs \
  --extractionmode direct \
  --ext direct_normal_20260615_660834 \
  --loglevel info

echo "Done. Synonyms saved alongside source triples."
