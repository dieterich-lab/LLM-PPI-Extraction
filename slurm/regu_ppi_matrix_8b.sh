#!/bin/bash
#
# Regulatome PPI extraction matrix — new codebase (2026)
#
# Matrix: [direct, nerrel] × [normal, neg, pos, negpos, dynex_k=3, lookup, ensemble=5, tot]
# = 16 combinations, but direct+lookup is auto-upgraded to nerrel by the parser
#   (see parser.py), so it is identical to nerrel+lookup → skipped (15 runs total).
#
# Fixed params: --model llama31 --chattype oneshot --data regulatome --target ppi --doclevel docs
# --noconfidence is default=True in the current parser.

#SBATCH --job-name=regu_ppi_matrix_8b
#SBATCH --output=/beegfs/prj/LINDA_LLM/outputs/slurm/regu_ppi_matrix_8b_%j.log
#SBATCH --partition=gpu
#SBATCH --gres=gpu:ampere:1
#SBATCH --nodelist=gpu-g4-1
#SBATCH --mem=60G

set -euo pipefail

OLLAMA_PORT="11438"
OLLAMA_LOG_DIR="/beegfs/prj/LINDA_LLM/outputs/slurm"
mkdir -p "$OLLAMA_LOG_DIR"

WORKDIR="/beegfs/prj/LINDA_LLM/scripts"
if [[ ! -f "$WORKDIR/extract.py" ]]; then
  echo "ERROR: scripts dir not found at $WORKDIR"
  exit 1
fi

RUN_TAG="20260615_${SLURM_JOB_ID:-local}"

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

OLLAMA_LOG="${OLLAMA_LOG_DIR}/ollama_regu_ppi_matrix_8b_${SLURM_JOB_ID:-local}.log"
ollama serve > "$OLLAMA_LOG" 2>&1 &
OLLAMA_PID=$!

if ! wait_for_ollama "http://127.0.0.1:${OLLAMA_PORT}/api/tags"; then
  echo "ERROR: Ollama did not become ready on port ${OLLAMA_PORT}"
  exit 1
fi
echo "Ollama ready on port ${OLLAMA_PORT}"

# Common flags for all runs
COMMON="--model llama31 --node local --port 38 --chattype oneshot --data regulatome --target ppi --doclevel docs --loglevel info --force_new --full_corpus"

run() {
  local label="$1"; shift
  echo ""
  echo "========================================="
  echo "RUN: $label  [$(date '+%Y-%m-%dT%H:%M:%S')]"
  echo "========================================="
  python -u extract.py $COMMON --ext "${label}_${RUN_TAG}" "$@"
}

# ── DIRECT extractions (7 runs) ───────────────────────────────────────────────
run "direct_normal"    --extractionmode direct
run "direct_neg"       --extractionmode direct --examples neg
run "direct_pos"       --extractionmode direct --examples pos
run "direct_negpos"    --extractionmode direct --examples negpos
run "direct_dynex3"    --extractionmode direct --dynex_k 3
# NOTE: direct+lookup is auto-upgraded to nerrel by the parser → covered below
run "direct_ensemble5" --extractionmode direct --ensemble 3
run "direct_tot"       --extractionmode direct --tot

# ── NERREL extractions (8 runs) ───────────────────────────────────────────────
run "nerrel_normal"    --extractionmode nerrel
run "nerrel_neg"       --extractionmode nerrel --examples neg
run "nerrel_pos"       --extractionmode nerrel --examples pos
run "nerrel_negpos"    --extractionmode nerrel --examples negpos
run "nerrel_dynex3"    --extractionmode nerrel --dynex_k 3
run "nerrel_lookup"    --extractionmode nerrel --lookup
run "nerrel_ensemble5" --extractionmode nerrel --ensemble 3
run "nerrel_tot"       --extractionmode nerrel --tot

echo ""
echo "All 15 runs completed. Tag: ${RUN_TAG}"
echo "Outputs: /beegfs/prj/LINDA_LLM/outputs/triples/regulatome/ppi/llama31/"
