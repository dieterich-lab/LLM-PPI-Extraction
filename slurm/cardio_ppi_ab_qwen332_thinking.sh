#!/bin/bash

#SBATCH --job-name=cardio_ppi_ab_qwen332
#SBATCH --output=/beegfs/prj/LINDA_LLM/outputs/slurm/cardio_ppi_ab_qwen332_%j.log
#SBATCH --partition=gpu
#SBATCH --gres=gpu:hopper:1
#SBATCH --nodelist=gpu-g5-1
#SBATCH --mem=60G

set -euo pipefail

module unload ollama || true
module load ollama/0.11.8

DEDICATED_INPUT_DIR="/prj/LINDA_LLM/outputs/parsed_papers/CardioPrior/ppi/10paper_dedicated_20260602"
if [[ ! -d "$DEDICATED_INPUT_DIR" ]]; then
  echo "ERROR: Dedicated input directory does not exist: $DEDICATED_INPUT_DIR"
  exit 1
fi

WORKDIR="${SLURM_SUBMIT_DIR:-}"
if [[ -z "$WORKDIR" || ! -f "$WORKDIR/extract.py" ]]; then
  if [[ -f "/prj/LINDA_LLM/scripts/extract.py" ]]; then
    WORKDIR="/prj/LINDA_LLM/scripts"
  elif [[ -f "/beegfs/prj/LINDA_LLM/scripts/extract.py" ]]; then
    WORKDIR="/beegfs/prj/LINDA_LLM/scripts"
  else
    echo "ERROR: Could not locate scripts workdir with extract.py"
    exit 1
  fi
fi

BASELINE_WORKDIR="/prj/LINDA_LLM/scripts_ab_baseline"
if [[ ! -f "$BASELINE_WORKDIR/extract.py" ]]; then
  echo "ERROR: Baseline worktree not found at $BASELINE_WORKDIR"
  exit 1
fi

MODEL="qwen332"        # maps to qwen3:32b in clients.py (native thinking model)
LOGLEVEL="${LOGLEVEL:-info}"
EXAMPLES="${EXAMPLES:-pos}"
CURRENT_OLLAMA_PORT="11439"
BASELINE_OLLAMA_PORT="11440"
CURRENT_PORT_SUFFIX="39"
BASELINE_PORT_SUFFIX="40"

SLURM_LOG_DIR="/beegfs/prj/LINDA_LLM/outputs/slurm"
mkdir -p "$SLURM_LOG_DIR"

RUN_TAG="20260608_${SLURM_JOB_ID:-local}"
CUR_EXT="cardioprior10_pos_cur_qwen332_think_${RUN_TAG}"
BASE_EXT="cardioprior10_pos_base_qwen332_think_${RUN_TAG}"

cleanup() {
  if [[ -n "${CURRENT_OLLAMA_PID:-}" ]] && kill -0 "$CURRENT_OLLAMA_PID" >/dev/null 2>&1; then
    kill "$CURRENT_OLLAMA_PID" >/dev/null 2>&1 || true
    wait "$CURRENT_OLLAMA_PID" >/dev/null 2>&1 || true
  fi
  if [[ -n "${BASE_OLLAMA_PID:-}" ]] && kill -0 "$BASE_OLLAMA_PID" >/dev/null 2>&1; then
    kill "$BASE_OLLAMA_PID" >/dev/null 2>&1 || true
    wait "$BASE_OLLAMA_PID" >/dev/null 2>&1 || true
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

# ── ARM A: CURRENT CODE ───────────────────────────────────────────────────────
echo "=== ARM A: CURRENT CODE (qwen3:32b thinking) ==="
CUR_OLLAMA_LOG="${SLURM_LOG_DIR}/ollama_cardio_ppi_ab_qwen332_think_current_${SLURM_JOB_ID:-local}.log"
export OLLAMA_HOST="127.0.0.1:${CURRENT_OLLAMA_PORT}"
export OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:-1h}"
export OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-1}"
export OLLAMA_CONTEXT_LENGTH="${OLLAMA_CONTEXT_LENGTH:-80000}"
export OLLAMA_DEBUG=1
unset MAX_CHARS

ollama serve > "$CUR_OLLAMA_LOG" 2>&1 &
CURRENT_OLLAMA_PID=$!

if ! wait_for_ollama "http://127.0.0.1:${CURRENT_OLLAMA_PORT}/api/tags"; then
  echo "ERROR: current-arm ollama server did not become ready"
  exit 1
fi

cd "$WORKDIR"
export LINDA_LLM_PAPER_PATH_OVERRIDE="$DEDICATED_INPUT_DIR"

python -u extract.py \
  --target ppi \
  --data cardio \
  --model "$MODEL" \
  --node local \
  --port "$CURRENT_PORT_SUFFIX" \
  --loglevel "$LOGLEVEL" \
  --examples "$EXAMPLES" \
  --num-shards 1 \
  --shard-index 0 \
  --ext "$CUR_EXT" \
  --force_new

kill "$CURRENT_OLLAMA_PID" >/dev/null 2>&1 || true
wait "$CURRENT_OLLAMA_PID" >/dev/null 2>&1 || true
unset CURRENT_OLLAMA_PID

TRIPLE_DIR="/beegfs/prj/LINDA_LLM/outputs/triples/cardio/ppi/${MODEL}/direct/oneshot/docs/pos_ex"
CUR_OUT="${TRIPLE_DIR}/triples_${CUR_EXT}.jsonl"
if [[ ! -f "$CUR_OUT" ]]; then
  echo "ERROR: current-arm output not found: $CUR_OUT"
  exit 1
fi
echo "Current arm done: $CUR_OUT"

# ── ARM B: BASELINE COMMIT ────────────────────────────────────────────────────
echo "=== ARM B: BASELINE (sep2025, qwen3:32b thinking) ==="
echo "Generating baseline BAML client code"
(cd "$BASELINE_WORKDIR/baml" && baml-cli generate)

BASE_OLLAMA_LOG="${SLURM_LOG_DIR}/ollama_cardio_ppi_ab_qwen332_think_baseline_${SLURM_JOB_ID:-local}.log"
export OLLAMA_HOST="0.0.0.0:${BASELINE_OLLAMA_PORT}"
export OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:-1h}"
export OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-1}"
export OLLAMA_CONTEXT_LENGTH="${OLLAMA_CONTEXT_LENGTH:-80000}"
export OLLAMA_DEBUG=1
unset MAX_CHARS

ollama serve > "$BASE_OLLAMA_LOG" 2>&1 &
BASE_OLLAMA_PID=$!

if ! wait_for_ollama "http://127.0.0.1:${BASELINE_OLLAMA_PORT}/api/tags"; then
  echo "ERROR: baseline-arm ollama server did not become ready"
  exit 1
fi

cd "$BASELINE_WORKDIR"
export LINDA_LLM_PAPER_PATH_OVERRIDE="$DEDICATED_INPUT_DIR"

python -u extract.py \
  --target ppi \
  --data cardio \
  --model "$MODEL" \
  --node g5 \
  --port "$BASELINE_PORT_SUFFIX" \
  --loglevel "$LOGLEVEL" \
  --examples "$EXAMPLES" \
  --ext "$BASE_EXT" \
  --force_new

kill "$BASE_OLLAMA_PID" >/dev/null 2>&1 || true
wait "$BASE_OLLAMA_PID" >/dev/null 2>&1 || true
unset BASE_OLLAMA_PID

BASE_OUT="${TRIPLE_DIR}/triples_${BASE_EXT}.jsonl"
if [[ ! -f "$BASE_OUT" ]]; then
  echo "ERROR: baseline-arm output not found: $BASE_OUT"
  exit 1
fi
echo "Baseline arm done: $BASE_OUT"
echo "Done."
