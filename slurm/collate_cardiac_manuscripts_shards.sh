#!/bin/bash

#SBATCH --job-name=collate_cardiac_md
#SBATCH --output=/beegfs/prj/LINDA_LLM/outputs/slurm/collate_cardiac_manuscripts_%j.log
#SBATCH --partition=medium
#SBATCH --mem=4G

set -euo pipefail

RESULTS_JSONL_BASE="${RESULTS_JSONL_BASE:-/beegfs/prj/LINDA_LLM/outputs/cardiac_filter/cardiac_manuscripts_results_llama31.jsonl}"
NUM_SHARDS="${NUM_SHARDS:-4}"

if [[ "$RESULTS_JSONL_BASE" == *.jsonl ]]; then
  MERGED_JSONL="${RESULTS_JSONL_BASE%.jsonl}_merged.jsonl"
  DEDUP_JSONL="${RESULTS_JSONL_BASE%.jsonl}_merged_dedup.jsonl"
else
  MERGED_JSONL="${RESULTS_JSONL_BASE}_merged.jsonl"
  DEDUP_JSONL="${RESULTS_JSONL_BASE}_merged_dedup.jsonl"
fi

mkdir -p "$(dirname "$MERGED_JSONL")"
: > "$MERGED_JSONL"

MISSING=0
for ((i = 0; i < NUM_SHARDS; i++)); do
  if [[ "$RESULTS_JSONL_BASE" == *.jsonl ]]; then
    SHARD_JSONL="${RESULTS_JSONL_BASE%.jsonl}_shard${i}.jsonl"
  else
    SHARD_JSONL="${RESULTS_JSONL_BASE}_shard${i}.jsonl"
  fi

  if [[ -f "$SHARD_JSONL" ]]; then
    cat "$SHARD_JSONL" >> "$MERGED_JSONL"
  else
    echo "WARNING: Missing shard output: $SHARD_JSONL"
    MISSING=1
  fi
done

/bin/python3 - "$MERGED_JSONL" "$DEDUP_JSONL" <<'PY'
import json
import sys

merged_path = sys.argv[1]
dedup_path = sys.argv[2]
seen = set()

with open(merged_path, "r", encoding="utf-8") as src, open(dedup_path, "w", encoding="utf-8") as dst:
    for line in src:
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        key = row.get("file_path")
        if not isinstance(key, str) or key in seen:
            continue
        seen.add(key)
        dst.write(json.dumps(row) + "\n")
PY

echo "Merged output: $MERGED_JSONL"
echo "Deduped output: $DEDUP_JSONL"
wc -l "$MERGED_JSONL" "$DEDUP_JSONL"

if [[ "$MISSING" -ne 0 ]]; then
  echo "WARNING: One or more shard files were missing during collation"
fi
