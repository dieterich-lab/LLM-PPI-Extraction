#!/bin/bash
#SBATCH --job-name=cur5_ll31regu_unc
#SBATCH --output=/beegfs/prj/LINDA_LLM/outputs/slurm/cur5_ll31regu_unc_%j.log
#SBATCH --partition=gpu
#SBATCH --gres=gpu:ampere:1
#SBATCH --nodelist=gpu-g4-1
#SBATCH --mem=32G

set -euo pipefail

cd /prj/LINDA_LLM/scripts
. ~/.venvs/test_linda/bin/activate

MODEL="${MODEL:-llama31regu}"
TARGET="${TARGET:-ppi}"
DATA="${DATA:-5curated}"
EXTRACTIONMODE="${EXTRACTIONMODE:-direct}"
CHATTYPE="${CHATTYPE:-oneshot}"
DOCLEVEL="${DOCLEVEL:-docs}"
PORT_SUFFIX="${PORT_SUFFIX:-34}"
OLLAMA_PORT="${OLLAMA_PORT:-11434}"
PAPER_DIR_OVERRIDE="${PAPER_DIR_OVERRIDE:-/prj/LINDA_LLM/outputs/parsed_papers/ppi/llama_parse/5curated}"
OLLAMA_LOG="/beegfs/prj/LINDA_LLM/outputs/slurm/ollama_cur5_ll31regu_unc_${SLURM_JOB_ID:-local}.log"
EXT="${EXT:-cur5_unc_llama31regu_${SLURM_JOB_ID:-local}}"

mkdir -p /beegfs/prj/LINDA_LLM/outputs/slurm

export OLLAMA_HOST="127.0.0.1:${OLLAMA_PORT}"
export OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:-1h}"
export OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-1}"
export OLLAMA_CONTEXT_LENGTH="${OLLAMA_CONTEXT_LENGTH:-128000}"
export LINDA_LLM_PAPER_PATH_OVERRIDE="$PAPER_DIR_OVERRIDE"
unset MAX_CHARS || true

ollama serve > "$OLLAMA_LOG" 2>&1 &
OLLAMA_PID=$!

cleanup() {
  if [[ -n "${OLLAMA_PID:-}" ]] && kill -0 "$OLLAMA_PID" >/dev/null 2>&1; then
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

echo "Running 5curated unconstraint extraction"
echo "model=$MODEL target=$TARGET data=$DATA ext=$EXT"
echo "paper_dir_override=$PAPER_DIR_OVERRIDE"
echo "node=$(hostname) ollama_host=$OLLAMA_HOST"
echo "start=$(date '+%Y-%m-%d %H:%M:%S')"

python -u extract.py \
  --model "$MODEL" \
  --target "$TARGET" \
  --data "$DATA" \
  --extractionmode "$EXTRACTIONMODE" \
  --chattype "$CHATTYPE" \
  --doclevel "$DOCLEVEL" \
  --node local \
  --port "$PORT_SUFFIX" \
  --loglevel off \
  --ext "$EXT"

echo "end=$(date '+%Y-%m-%d %H:%M:%S')"
