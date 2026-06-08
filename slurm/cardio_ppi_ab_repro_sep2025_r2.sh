#!/bin/bash

#SBATCH --job-name=cardio_ppi_ab_r2
#SBATCH --output=/beegfs/prj/LINDA_LLM/outputs/slurm/cardio_ppi_ab_r2_%j.log
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

MODEL="${MODEL:-llama33}"
LOGLEVEL="${LOGLEVEL:-info}"
EXAMPLES="${EXAMPLES:-pos}"
CURRENT_OLLAMA_PORT="11439"
BASELINE_OLLAMA_PORT="11440"
CURRENT_PORT_SUFFIX="39"
BASELINE_PORT_SUFFIX="40"

SLURM_LOG_DIR="/beegfs/prj/LINDA_LLM/outputs/slurm"
mkdir -p "$SLURM_LOG_DIR"

RUN_TAG="20260608_${SLURM_JOB_ID:-local}"
CUR_EXT="cardioprior10_pos_cur_${RUN_TAG}"
BASE_EXT="cardioprior10_pos_base_${RUN_TAG}"

TRIPLE_DIR="/beegfs/prj/LINDA_LLM/outputs/triples/cardio/ppi/${MODEL}/direct/oneshot/docs/pos_ex"

# Previous run results (for determinism comparison)
PREV_CUR_OUT="${TRIPLE_DIR}/triples_cardioprior10_pos_20260602.jsonl"
PREV_BASE_OUT="${TRIPLE_DIR}/triples_cardioprior10_pos_ab_baseline_20250603.jsonl"

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

# ── ARM A: CURRENT CODE ───────────────────────────────────────────────────────
echo "=== ARM A: CURRENT CODE ==="
CUR_OLLAMA_LOG="${SLURM_LOG_DIR}/ollama_cardio_ppi_ab_r2_current_${SLURM_JOB_ID:-local}.log"
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
. ~/.venvs/test_linda/bin/activate
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

CUR_OUT="${TRIPLE_DIR}/triples_${CUR_EXT}.jsonl"
if [[ ! -f "$CUR_OUT" ]]; then
  echo "ERROR: current-arm output not found: $CUR_OUT"
  exit 1
fi

# ── ARM B: BASELINE COMMIT ────────────────────────────────────────────────────
echo "=== ARM B: BASELINE (sep2025) ==="
echo "Generating baseline BAML client code"
(cd "$BASELINE_WORKDIR/baml" && baml-cli generate)

BASE_OLLAMA_LOG="${SLURM_LOG_DIR}/ollama_cardio_ppi_ab_r2_baseline_${SLURM_JOB_ID:-local}.log"
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

cd "$WORKDIR"

# ── EVALUATION ────────────────────────────────────────────────────────────────
echo "=== EVALUATION ==="
REPORT_PATH="/beegfs/prj/LINDA_LLM/outputs/evaluations/cardio_ppi_ab_r2_${SLURM_JOB_ID:-local}.md"
mkdir -p "$(dirname "$REPORT_PATH")"

export CUR_OUT BASE_OUT PREV_CUR_OUT PREV_BASE_OUT REPORT_PATH

python - <<'PY'
import json
import os
from pathlib import Path

cur_out      = Path(os.environ["CUR_OUT"])
base_out     = Path(os.environ["BASE_OUT"])
prev_cur_out = Path(os.environ["PREV_CUR_OUT"])
prev_base_out= Path(os.environ["PREV_BASE_OUT"])
report_path  = Path(os.environ["REPORT_PATH"])


def norm_triple(t):
    h  = str(t.get("head",     "")).strip().lower()
    r  = str(t.get("relation", "")).strip().upper()
    ta = str(t.get("tail",     "")).strip().lower()
    return (h, r, ta) if (h and r and ta) else None


def read_map(path):
    """Return {stem -> set(normalised triples)}."""
    m = {}
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            fn  = Path(obj.get("filename", "")).stem
            if not fn:
                continue
            triples = set()
            for response in obj.get("responses", []):
                for t in response:
                    nt = norm_triple(t)
                    if nt:
                        triples.add(nt)
            m[fn] = triples
    return m


def jaccard(a, b):
    u = a | b
    return (len(a & b) / len(u)) if u else 1.0


def per_doc_table(map_a, map_b, label_a, label_b):
    rows = []
    all_docs = sorted(set(map_a) | set(map_b))
    rows.append(f"| doc | {label_a} | {label_b} | intersection | only_{label_a} | only_{label_b} | jaccard |")
    rows.append("|---|---:|---:|---:|---:|---:|---:|")
    for d in all_docs:
        a = map_a.get(d, set())
        b = map_b.get(d, set())
        i = a & b
        rows.append(
            f"| {d} | {len(a)} | {len(b)} | {len(i)} | {len(a - b)} | {len(b - a)} | {jaccard(a, b):.4f} |"
        )
    return rows


def global_stats(map_a, map_b, label_a, label_b):
    ua = set().union(*map_a.values()) if map_a else set()
    ub = set().union(*map_b.values()) if map_b else set()
    rows = [
        f"- {label_a} unique triples: {len(ua)}",
        f"- {label_b} unique triples: {len(ub)}",
        f"- Intersection: {len(ua & ub)}",
        f"- Only {label_a}: {len(ua - ub)}",
        f"- Only {label_b}: {len(ub - ua)}",
        f"- Jaccard: {jaccard(ua, ub):.4f}",
    ]
    return rows


cur       = read_map(cur_out)
base      = read_map(base_out)
prev_cur  = read_map(prev_cur_out)  if prev_cur_out.exists()  else {}
prev_base = read_map(prev_base_out) if prev_base_out.exists() else {}

lines = []
lines.append("# Cardio PPI A/B Repro Report (r2)")
lines.append("")
lines.append(f"- Current output:      {cur_out}")
lines.append(f"- Baseline output:     {base_out}")
lines.append(f"- Prev current output: {prev_cur_out}")
lines.append(f"- Prev baseline output:{prev_base_out}")
lines.append("")

# ── Section 1: A/B overlap (new current vs new baseline) ────────────────────
lines.append("## 1. A/B Overlap: current vs baseline (new runs)")
lines.append("")
lines += global_stats(cur, base, "current", "baseline")
lines.append("")
lines += per_doc_table(cur, base, "current", "baseline")
lines.append("")

# ── Section 2: Determinism — current ────────────────────────────────────────
lines.append("## 2. Determinism — current code (new vs job-659018)")
lines.append("")
if prev_cur:
    lines += global_stats(cur, prev_cur, "new_cur", "prev_cur_659018")
    lines.append("")
    lines += per_doc_table(cur, prev_cur, "new_cur", "prev_cur_659018")
else:
    lines.append("_Previous current output not found — skipped._")
lines.append("")

# ── Section 3: Determinism — baseline ───────────────────────────────────────
lines.append("## 3. Determinism — baseline code (new vs job-659019)")
lines.append("")
if prev_base:
    lines += global_stats(base, prev_base, "new_base", "prev_base_659019")
    lines.append("")
    lines += per_doc_table(base, prev_base, "new_base", "prev_base_659019")
else:
    lines.append("_Previous baseline output not found — skipped._")
lines.append("")

report_path.write_text("\n".join(lines) + "\n")
print("Wrote", report_path)

# Also print key numbers to stdout
cur_u  = set().union(*cur.values())  if cur  else set()
base_u = set().union(*base.values()) if base else set()
print(f"A/B Jaccard (current vs baseline): {jaccard(cur_u, base_u):.4f}")

if prev_cur:
    prev_cur_u = set().union(*prev_cur.values())
    print(f"Determinism Jaccard (current new vs 659018): {jaccard(cur_u, prev_cur_u):.4f}")
if prev_base:
    prev_base_u = set().union(*prev_base.values())
    print(f"Determinism Jaccard (baseline new vs 659019): {jaccard(base_u, prev_base_u):.4f}")
PY

echo "Done. Report: ${REPORT_PATH}"
