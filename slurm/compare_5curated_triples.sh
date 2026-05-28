#!/bin/bash
#SBATCH --job-name=cur5_triple_cmp
#SBATCH --output=/beegfs/prj/LINDA_LLM/outputs/slurm/cur5_triple_cmp_%j.log
#SBATCH --partition=medium
#SBATCH --mem=8G

set -euo pipefail

cd /prj/LINDA_LLM/scripts
. ~/.venvs/test_linda/bin/activate

JOB_70="${JOB_70:?JOB_70 is required}"
JOB_8="${JOB_8:?JOB_8 is required}"

FILE_70="/beegfs/prj/LINDA_LLM/outputs/triples/5curated/ppi/llama33/direct/oneshot/docs/triples_cur5_unc_llama33_${JOB_70}.jsonl"
FILE_8="/beegfs/prj/LINDA_LLM/outputs/triples/5curated/ppi/llama31regu/direct/oneshot/docs/triples_cur5_unc_llama31regu_${JOB_8}.jsonl"

OUT_DIR="/beegfs/prj/LINDA_LLM/outputs/evaluations"
mkdir -p "$OUT_DIR"
OUT_JSON="$OUT_DIR/cur5_triple_compare_${JOB_70}_vs_${JOB_8}.json"
export FILE_70 FILE_8 OUT_JSON

python - << 'PY'
import json
import os
from pathlib import Path

file_70 = Path(os.environ["FILE_70"])
file_8 = Path(os.environ["FILE_8"])
out_json = Path(os.environ["OUT_JSON"])

if not file_70.exists():
    raise FileNotFoundError(f"Missing 70B output: {file_70}")
if not file_8.exists():
    raise FileNotFoundError(f"Missing 8B output: {file_8}")


def load_rows(path: Path):
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def canonical_triple(t):
    h = str(t.get("head", "")).strip().lower()
    r = str(t.get("relation", "")).strip().upper()
    ta = str(t.get("tail", "")).strip().lower()
    c = str(t.get("confidence", "")).strip().lower()
    return (h, r, ta, c)


def flatten(row):
    out = []
    for resp in row.get("responses", []):
        if isinstance(resp, list):
            for t in resp:
                if isinstance(t, dict):
                    out.append(t)
    return out


rows70 = load_rows(file_70)
rows8 = load_rows(file_8)

set70 = set()
set8 = set()
by_file70 = {}
by_file8 = {}

for row in rows70:
    fname = Path(str(row.get("filename", ""))).name
    triples = {canonical_triple(t) for t in flatten(row)}
    by_file70[fname] = sorted(list(triples))
    set70.update(triples)

for row in rows8:
    fname = Path(str(row.get("filename", ""))).name
    triples = {canonical_triple(t) for t in flatten(row)}
    by_file8[fname] = sorted(list(triples))
    set8.update(triples)

inter = set70 & set8
only70 = set70 - set8
only8 = set8 - set70
union = set70 | set8
jaccard = (len(inter) / len(union)) if union else 0.0

result = {
    "file_70": str(file_70),
    "file_8": str(file_8),
    "docs_70": len(rows70),
    "docs_8": len(rows8),
    "triples_70_unique": len(set70),
    "triples_8_unique": len(set8),
    "intersection_unique": len(inter),
    "only_70_unique": len(only70),
    "only_8_unique": len(only8),
    "jaccard": round(jaccard, 6),
    "sample_only_70": [list(x) for x in sorted(only70)[:20]],
    "sample_only_8": [list(x) for x in sorted(only8)[:20]],
}

with out_json.open("w") as f:
    json.dump(result, f, indent=2)

print("=== 5CURATED TRIPLE COMPARISON ===")
print(f"docs_70={result['docs_70']} docs_8={result['docs_8']}")
print(f"triples_70_unique={result['triples_70_unique']}")
print(f"triples_8_unique={result['triples_8_unique']}")
print(f"intersection_unique={result['intersection_unique']}")
print(f"only_70_unique={result['only_70_unique']}")
print(f"only_8_unique={result['only_8_unique']}")
print(f"jaccard={result['jaccard']}")
print(f"saved_json={out_json}")
PY
