#!/bin/bash
#SBATCH --job-name=analyze_hf_bench
#SBATCH --output=/beegfs/prj/LINDA_LLM/outputs/slurm/analyze_hf_bench_%j.log
#SBATCH --partition=gpu
#SBATCH --nodelist=gpu-g4-1
#SBATCH --gres=gpu:ampere:1
#SBATCH --mem=8G

set -euo pipefail

cd /prj/LINDA_LLM
. ~/.venvs/test_linda/bin/activate

JOB_1="${JOB_1:-658390}"
JOB_2="${JOB_2:-658391}"
JOB_3="${JOB_3:-658392}"
MAX_CHARS_1="${MAX_CHARS_1:-20000}"
MAX_CHARS_2="${MAX_CHARS_2:-60000}"
MAX_CHARS_3="${MAX_CHARS_3:-0}"
MODEL_ALIAS="${MODEL_ALIAS:-llama31regu}"
TOTAL_DOCS_ETA="${TOTAL_DOCS_ETA:-36000}"

echo "Analyzing HF benchmark jobs"
echo "JOB_1=$JOB_1 MAX_CHARS_1=$MAX_CHARS_1"
echo "JOB_2=$JOB_2 MAX_CHARS_2=$MAX_CHARS_2"
echo "JOB_3=$JOB_3 MAX_CHARS_3=$MAX_CHARS_3"
echo "MODEL_ALIAS=$MODEL_ALIAS"
echo "TOTAL_DOCS_ETA=$TOTAL_DOCS_ETA"

~/.venvs/test_linda/bin/python - << 'PY'
import csv
import json
import os
import re
import subprocess
from pathlib import Path

job_ids = [os.environ["JOB_1"], os.environ["JOB_2"], os.environ["JOB_3"]]
max_chars_list = [
    os.environ["MAX_CHARS_1"],
    os.environ["MAX_CHARS_2"],
    os.environ["MAX_CHARS_3"],
]
model_alias = os.environ["MODEL_ALIAS"]
total_docs_eta = int(os.environ["TOTAL_DOCS_ETA"])

sacct_cmd = [
    "sacct",
    "-j",
    ",".join(job_ids),
    "--format=JobIDRaw,State,ExitCode,ElapsedRaw",
    "-n",
    "-P",
]
res = subprocess.run(sacct_cmd, check=True, capture_output=True, text=True)

elapsed_by_job = {}
state_by_job = {}
for line in res.stdout.strip().splitlines():
    parts = line.split("|")
    if len(parts) < 4:
        continue
    jobid_raw, state, exit_code, elapsed_raw = parts[:4]
    if "." in jobid_raw:
        continue
    if jobid_raw in job_ids:
        state_by_job[jobid_raw] = f"{state} ({exit_code})"
        try:
            elapsed_by_job[jobid_raw] = float(elapsed_raw)
        except Exception:
            elapsed_by_job[jobid_raw] = float("nan")

print("=== HF BENCH AGGREGATE SUMMARY ===")
print("max_chars\tjob_id\tstate\tdocs\tsec_doc\teta_days_36k\tjson_valid_pct\tjson_invalid_pct")

for max_chars, job_id in zip(max_chars_list, job_ids):
    out_jsonl = Path(f"/beegfs/prj/LINDA_LLM/outputs/triples/cardio/ppi/{model_alias}/hf_bench/triples_hf_bench_{job_id}.jsonl")
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
        else:
            invalid += 1

    elapsed = elapsed_by_job.get(job_id, float("nan"))
    sec_doc = (elapsed / n_docs) if (n_docs and elapsed == elapsed) else float("nan")
    eta_days = ((sec_doc * total_docs_eta) / 86400.0) if sec_doc == sec_doc else float("nan")
    valid_pct = (100.0 * valid / n_docs) if n_docs else 0.0
    invalid_pct = (100.0 * invalid / n_docs) if n_docs else 0.0
    state = state_by_job.get(job_id, "unknown")

    print(
        f"{max_chars}\t{job_id}\t{state}\t{n_docs}\t{sec_doc:.2f}\t{eta_days:.2f}\t{valid_pct:.2f}\t{invalid_pct:.2f}"
    )
PY
