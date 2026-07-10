#!/bin/bash
#
# RegulaTome PPI extraction matrix — fine-tuned Llama 3.3 70B (RegulaTome PPI)
#
# Matrix: [direct, nerrel] × [normal, neg, pos, negpos, dynex_k=3, lookup, ensemble=5, tot]
# = 15 runs on full corpus (train+dev+test = 1591 docs).
#
# Prerequisites:
#   - ollama_import_llama33_regu_ppi.sh must have been run first
#   - Model 'llama3.3:70b-regulatome-ppi' must exist in Ollama
#
# Fixed params: --model llama33regu --chattype oneshot --data regulatome --target ppi --doclevel docs --full_corpus

#SBATCH --job-name=regu_ppi_matrix_33regu
#SBATCH --output=../outputs/slurm/regu_ppi_matrix_33regu_%j.log
#SBATCH --partition=gpu
#SBATCH --gres=gpu:hopper:1
#SBATCH --nodelist=gpu-g5-1
#SBATCH --mem=100G

set -euo pipefail

# ── Load .env configuration ────────────────────────────────────────────
if [[ -f "${SLURM_SUBMIT_DIR:-.}/.env" ]]; then
  set -a; source "${SLURM_SUBMIT_DIR:-.}/.env"; set +a
fi

# ── Project root discovery ─────────────────────────────────────────────
if [[ -n "${LINDA_LLM_PROJECT_ROOT:-}" ]]; then
  WORKDIR="$LINDA_LLM_PROJECT_ROOT/scripts"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  WORKDIR="$SLURM_SUBMIT_DIR"
else
  echo "ERROR: Set LINDA_LLM_PROJECT_ROOT or submit from scripts/."
  exit 1
fi

cd "$WORKDIR"
VENV="${LINDA_LLM_PYTHON_VENV:-${HOME}/.venvs/test_linda}"
. "$VENV/bin/activate"

OLLAMA_PORT="11441"
OLLAMA_LOG_DIR="${LINDA_LLM_SLURM_LOG_DIR:-$WORKDIR/../outputs/slurm}"
mkdir -p "$OLLAMA_LOG_DIR"

RUN_TAG="${SLURM_JOB_ID:-local}"

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

export OLLAMA_HOST="127.0.0.1:${OLLAMA_PORT}"
export OLLAMA_KEEP_ALIVE="4h"
export OLLAMA_NUM_PARALLEL=1
export OLLAMA_CONTEXT_LENGTH=80000
export OLLAMA_DEBUG=1
unset MAX_CHARS

OLLAMA_LOG="${OLLAMA_LOG_DIR}/ollama_regu_ppi_matrix_33regu_${SLURM_JOB_ID:-local}.log"
ollama serve > "$OLLAMA_LOG" 2>&1 &
OLLAMA_PID=$!

if ! wait_for_ollama "http://127.0.0.1:${OLLAMA_PORT}/api/tags"; then
  echo "ERROR: Ollama did not become ready on port ${OLLAMA_PORT}"
  exit 1
fi
echo "Ollama ready on port ${OLLAMA_PORT}"

# Common flags for all runs
COMMON="--model llama33regu --node local --port 41 --chattype oneshot --data regulatome --target ppi --doclevel docs --loglevel info --force_new --full_corpus"

run() {
  local label="$1"; shift
  echo ""
  echo "========================================="
  echo "RUN: $label  [$(date '+%Y-%m-%dT%H:%M:%S')]"
  echo "========================================="
  python -u extract.py $COMMON --ext "${label}_${RUN_TAG}" "$@"
}

# ── DIRECT extractions (7 runs) ───────────────────────────────────────
run "direct_normal"    --extractionmode direct
run "direct_neg"       --extractionmode direct --examples neg
run "direct_pos"       --extractionmode direct --examples pos
run "direct_negpos"    --extractionmode direct --examples negpos
run "direct_dynex3"    --extractionmode direct --dynex_k 3
run "direct_ensemble5" --extractionmode direct --ensemble 3
run "direct_tot"       --extractionmode direct --tot

# ── NERREL extractions (8 runs) ───────────────────────────────────────
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
echo "Outputs: ${LINDA_LLM_TRIPLES_ROOT:-$WORKDIR/../outputs/triples}/regulatome/ppi/llama33regu/"
