#!/bin/bash

#SBATCH --job-name=cardio_ppi_ab_sep2025
#SBATCH --output=/beegfs/prj/LINDA_LLM/outputs/slurm/cardio_ppi_ab_sep2025_%j.log
#SBATCH --partition=gpu
#SBATCH --gres=gpu:hopper:1
#SBATCH --nodelist=gpu-g5-1
#SBATCH --mem=60G

set -euo pipefail

BASELINE_COMMIT="dfb60979a99f780c21a922c64a3e992588681195" # last commit before Sep-2025 window
MODEL="${MODEL:-llama33}"
LOGLEVEL="${LOGLEVEL:-info}"
EXAMPLES="${EXAMPLES:-pos}"
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

cd "$WORKDIR"

if ! command -v ollama >/dev/null 2>&1; then
  echo "ERROR: ollama command not found"
  exit 1
fi

module unload ollama || true
module load ollama/0.11.8

. ~/.venvs/test_linda/bin/activate

RUN_TAG="absep2025_${SLURM_JOB_ID:-local}"
CUR_EXT="cardioprior10_pos_${RUN_TAG}_current"
BASE_EXT="cardioprior10_pos_${RUN_TAG}_baseline"

TMP_ROOT="/tmp/cardio_ab_repro_${SLURM_JOB_ID:-local}"
BASELINE_WT="${TMP_ROOT}/scripts_baseline"
mkdir -p "$TMP_ROOT"

SLURM_LOG_DIR="/beegfs/prj/LINDA_LLM/outputs/slurm"
mkdir -p "$SLURM_LOG_DIR"

CURRENT_OLLAMA_PORT="11439"
BASELINE_OLLAMA_PORT="11440"
CURRENT_PORT_SUFFIX="39"
BASELINE_PORT_SUFFIX="40"

cleanup() {
  if [[ -n "${CURRENT_OLLAMA_PID:-}" ]] && kill -0 "$CURRENT_OLLAMA_PID" >/dev/null 2>&1; then
    kill "$CURRENT_OLLAMA_PID" >/dev/null 2>&1 || true
    wait "$CURRENT_OLLAMA_PID" >/dev/null 2>&1 || true
  fi
  if [[ -n "${BASE_OLLAMA_PID:-}" ]] && kill -0 "$BASE_OLLAMA_PID" >/dev/null 2>&1; then
    kill "$BASE_OLLAMA_PID" >/dev/null 2>&1 || true
    wait "$BASE_OLLAMA_PID" >/dev/null 2>&1 || true
  fi
  if [[ -d "$BASELINE_WT" ]]; then
    git worktree remove --force "$BASELINE_WT" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

wait_for_ollama() {
  local port="$1"
  local ready=0
  for _ in {1..90}; do
    if curl --silent --fail --output /dev/null "http://127.0.0.1:${port}/api/tags"; then
      ready=1
      break
    fi
    sleep 1
  done
  if [[ "$ready" -ne 1 ]]; then
    return 1
  fi
  return 0
}

echo "=== ARM A: CURRENT CODE ==="
CUR_OLLAMA_LOG="${SLURM_LOG_DIR}/ollama_cardio_ppi_ab_current_${SLURM_JOB_ID:-local}.log"
export OLLAMA_HOST="127.0.0.1:${CURRENT_OLLAMA_PORT}"
export OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:-1h}"
export OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-1}"
export OLLAMA_CONTEXT_LENGTH="${OLLAMA_CONTEXT_LENGTH:-80000}"
export OLLAMA_DEBUG=1
unset MAX_CHARS

ollama serve > "$CUR_OLLAMA_LOG" 2>&1 &
CURRENT_OLLAMA_PID=$!

if ! wait_for_ollama "$CURRENT_OLLAMA_PORT"; then
  echo "ERROR: current-arm ollama server did not become ready"
  exit 1
fi

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

CUR_OUT="/beegfs/prj/LINDA_LLM/outputs/triples/cardio/ppi/${MODEL}/direct/oneshot/docs/pos_ex/triples_${CUR_EXT}.jsonl"
if [[ ! -f "$CUR_OUT" ]]; then
  echo "ERROR: current-arm output not found: $CUR_OUT"
  exit 1
fi

echo "=== ARM B: BASELINE COMMIT ${BASELINE_COMMIT} ==="
if [[ -d "$BASELINE_WT" ]]; then
  git worktree remove --force "$BASELINE_WT" >/dev/null 2>&1 || true
fi
git worktree add "$BASELINE_WT" "$BASELINE_COMMIT" >/dev/null

# Minimal compatibility shim: support dedicated input override in baseline documents.py.
if ! grep -q 'LINDA_LLM_PAPER_PATH_OVERRIDE' "$BASELINE_WT/documents.py"; then
  perl -0777 -i -pe 's/(elif args.data == "5curated":\n\s+_paper_paths = Path\(\n\s+f"\/beegfs\/prj\/LINDA_LLM\/outputs\/parsed_papers\/ppi\/\{args\.parser\}\/5curated\/"\n\s+\)\n)/$1    override_path = os.environ.get("LINDA_LLM_PAPER_PATH_OVERRIDE")\n    if override_path:\n        _paper_paths = Path(override_path)\n/s' "$BASELINE_WT/documents.py"
fi

BASE_OLLAMA_LOG="${SLURM_LOG_DIR}/ollama_cardio_ppi_ab_baseline_${SLURM_JOB_ID:-local}.log"
export OLLAMA_HOST="0.0.0.0:${BASELINE_OLLAMA_PORT}"
export OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:-1h}"
export OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-1}"
export OLLAMA_CONTEXT_LENGTH="${OLLAMA_CONTEXT_LENGTH:-80000}"
export OLLAMA_DEBUG=1
unset MAX_CHARS

ollama serve > "$BASE_OLLAMA_LOG" 2>&1 &
BASE_OLLAMA_PID=$!

if ! wait_for_ollama "$BASELINE_OLLAMA_PORT"; then
  echo "ERROR: baseline-arm ollama server did not become ready"
  exit 1
fi

cd "$BASELINE_WT"
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

BASE_OUT="/beegfs/prj/LINDA_LLM/outputs/triples/cardio/ppi/${MODEL}/direct/oneshot/docs/pos_ex/triples_${BASE_EXT}.jsonl"
if [[ ! -f "$BASE_OUT" ]]; then
  echo "ERROR: baseline-arm output not found: $BASE_OUT"
  exit 1
fi

cd "$WORKDIR"

echo "=== A/B OVERLAP REPORT ==="
REPORT_PATH="/beegfs/prj/LINDA_LLM/outputs/evaluations/cardio_ppi_ab_sep2025_${SLURM_JOB_ID:-local}.md"
mkdir -p "$(dirname "$REPORT_PATH")"
export CUR_OUT BASE_OUT REPORT_PATH

python - <<'PY'
import json
import os
from pathlib import Path

cur_out = Path(os.environ["CUR_OUT"])
base_out = Path(os.environ["BASE_OUT"])
report_path = Path(os.environ["REPORT_PATH"])

def norm_triples(entry):
    triples = set()
    for response in entry.get("responses", []):
        for t in response:
            h = str(t.get("head", "")).strip().lower()
            r = str(t.get("relation", "")).strip().upper()
            ta = str(t.get("tail", "")).strip().lower()
            if h and r and ta:
                triples.add((h, r, ta))
    return triples

def read_map(path):
    m = {}
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            fn = Path(obj.get("filename", "")).stem
            if fn:
                m[fn] = norm_triples(obj)
    return m

cur = read_map(cur_out)
base = read_map(base_out)
all_docs = sorted(set(cur) | set(base))

lines = []
lines.append("# Cardio PPI A/B Repro Report")
lines.append("")
lines.append(f"- Current output: {cur_out}")
lines.append(f"- Baseline output: {base_out}")
lines.append(f"- Docs (union): {len(all_docs)}")
lines.append("")

cur_union = set().union(*cur.values()) if cur else set()
base_union = set().union(*base.values()) if base else set()
inter = cur_union & base_union
only_cur = cur_union - base_union
only_base = base_union - cur_union

jaccard = (len(inter) / len(cur_union | base_union)) if (cur_union or base_union) else 1.0
lines.append("## Global")
lines.append(f"- Current unique triples: {len(cur_union)}")
lines.append(f"- Baseline unique triples: {len(base_union)}")
lines.append(f"- Intersection: {len(inter)}")
lines.append(f"- Only current: {len(only_cur)}")
lines.append(f"- Only baseline: {len(only_base)}")
lines.append(f"- Jaccard: {jaccard:.4f}")
lines.append("")

lines.append("## Per-doc")
lines.append("| doc | current | baseline | intersection | only_current | only_baseline | jaccard |")
lines.append("|---|---:|---:|---:|---:|---:|---:|")
for d in all_docs:
    c = cur.get(d, set())
    b = base.get(d, set())
    i = c & b
    u = c | b
    j = (len(i) / len(u)) if u else 1.0
    lines.append(
        f"| {d} | {len(c)} | {len(b)} | {len(i)} | {len(c - b)} | {len(b - c)} | {j:.4f} |"
    )

report_path.write_text("\n".join(lines) + "\n")
print("Wrote", report_path)
print("Global Jaccard:", f"{jaccard:.4f}")
PY

echo "Done. Report: ${REPORT_PATH}"
