#!/bin/bash
#SBATCH --job-name=bench_hf_ppi
#SBATCH --output=/beegfs/prj/LINDA_LLM/outputs/slurm/bench_hf_ppi_%j.log
#SBATCH --partition=gpu
#SBATCH --gres=gpu:ampere:1
#SBATCH --nodelist=gpu-g4-1
#SBATCH --mem=64G

set -euo pipefail

cd /prj/LINDA_LLM/scripts
. ~/.venvs/test_linda/bin/activate

MODEL_ALIAS="${MODEL_ALIAS:-llama31regu}"
INPUT_DIR="${INPUT_DIR:-/beegfs/prj/LINDA_LLM/Cardiac_Manuscripts_test10}"
MAX_CHARS="${MAX_CHARS:-20000}"
N_DOCS="${N_DOCS:-10}"
START_IDX="${START_IDX:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
TOTAL_DOCS_ETA="${TOTAL_DOCS_ETA:-36000}"
TARGET="${TARGET:-ppi}"
EXTRACTIONMODE="${EXTRACTIONMODE:-direct}"
CHATTYPE="${CHATTYPE:-oneshot}"

SLURM_LOG_DIR="/beegfs/prj/LINDA_LLM/outputs/slurm"
OUT_DIR="/beegfs/prj/LINDA_LLM/outputs/triples/cardio/ppi/${MODEL_ALIAS}/hf_bench"
mkdir -p "$SLURM_LOG_DIR" "$OUT_DIR"

OUT_JSONL="${OUT_DIR}/triples_hf_bench_${SLURM_JOB_ID:-local}.jsonl"

echo "Starting HF benchmark"
echo "Model alias: $MODEL_ALIAS"
echo "Input dir: $INPUT_DIR"
echo "Target: $TARGET"
echo "Extraction mode: $EXTRACTIONMODE"
echo "Chat type: $CHATTYPE"
echo "MAX_CHARS: $MAX_CHARS"
echo "N_DOCS: $N_DOCS"
echo "START_IDX: $START_IDX"
echo "MAX_NEW_TOKENS: $MAX_NEW_TOKENS"
echo "TOTAL_DOCS_ETA: $TOTAL_DOCS_ETA"
echo "Output: $OUT_JSONL"
echo "Start timestamp: $(date '+%Y-%m-%d %H:%M:%S')"

START_TS=$(date +%s)

~/.venvs/test_linda/bin/python hf_extract_ppi.py \
  --model "$MODEL_ALIAS" \
  --input-dir "$INPUT_DIR" \
  --output-jsonl "$OUT_JSONL" \
  --max-chars "$MAX_CHARS" \
  --start "$START_IDX" \
  --limit "$N_DOCS" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --target "$TARGET" \
  --extractionmode "$EXTRACTIONMODE" \
  --chattype "$CHATTYPE"

END_TS=$(date +%s)
DURATION=$((END_TS - START_TS))
export OUT_JSONL DURATION TOTAL_DOCS_ETA

python - << 'PY'
import json
import os
import re
from pathlib import Path

out_jsonl = Path(os.environ["OUT_JSONL"])
duration = float(os.environ["DURATION"])
total_docs_eta = int(os.environ["TOTAL_DOCS_ETA"])

rows = []
if out_jsonl.exists():
    with out_jsonl.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                pass

n_docs = len(rows)
valid = 0
invalid = 0
n_nonempty_triples = 0

for row in rows:
    raw = str(row.get("raw_output", ""))
    m = re.search(r"\{[\s\S]*\}", raw)
    if not m:
        invalid += 1
        continue
    try:
        parsed = json.loads(m.group(0))
    except Exception:
        invalid += 1
        continue

    triples = parsed.get("triples") if isinstance(parsed, dict) else None
    if isinstance(triples, list):
        valid += 1
        if len(triples) > 0:
            n_nonempty_triples += 1
    else:
        invalid += 1

sec_per_doc = duration / n_docs if n_docs else float("inf")
valid_pct = (100.0 * valid / n_docs) if n_docs else 0.0
invalid_pct = (100.0 * invalid / n_docs) if n_docs else 0.0
eta_seconds = sec_per_doc * total_docs_eta if n_docs else float("inf")
eta_hours = eta_seconds / 3600 if n_docs else float("inf")
eta_days = eta_hours / 24 if n_docs else float("inf")

print("=== HF BENCH SUMMARY ===")
print(f"docs_processed={n_docs}")
print(f"duration_seconds={duration:.2f}")
print(f"seconds_per_doc={sec_per_doc:.2f}")
print(f"eta_docs={total_docs_eta}")
print(f"eta_seconds={eta_seconds:.2f}")
print(f"eta_hours={eta_hours:.2f}")
print(f"eta_days={eta_days:.2f}")
print(f"json_valid_outputs={valid}")
print(f"json_invalid_outputs={invalid}")
print(f"json_valid_pct={valid_pct:.2f}")
print(f"json_invalid_pct={invalid_pct:.2f}")
print(f"nonempty_triple_outputs={n_nonempty_triples}")
print(f"output_jsonl={out_jsonl}")
PY

echo "End timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Total duration seconds: $DURATION"
