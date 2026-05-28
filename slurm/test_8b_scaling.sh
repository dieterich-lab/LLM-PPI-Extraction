#!/bin/bash

#SBATCH --job-name=test_8b_scaling
#SBATCH --output=/beegfs/prj/LINDA_LLM/outputs/slurm/test_8b_scaling_%j.log
#SBATCH --partition=gpu
#SBATCH --gres=gpu:hopper:1
#SBATCH --nodelist=gpu-g5-1
#SBATCH --mem=20G

set -euo pipefail

cd /prj/LINDA_LLM/scripts
. ~/.venvs/test_linda/bin/activate

# Parse arguments
MAX_CHARS="${1:-20000}"

export OLLAMA_HOST="127.0.0.1:11434"
export OLLAMA_KEEP_ALIVE="1h"
export OLLAMA_NUM_PARALLEL="1"
export OLLAMA_CONTEXT_LENGTH="128000"
export OLLAMA_DEBUG=1

OLLAMA_LOG="/beegfs/prj/LINDA_LLM/outputs/slurm/ollama_test_8b_${SLURM_JOB_ID}_${MAX_CHARS}.log"

# Start ollama
ollama serve > "$OLLAMA_LOG" 2>&1 &
OLLAMA_PID=$!

cleanup() {
  if [[ -n "${OLLAMA_PID:-}" ]] && kill -0 "$OLLAMA_PID" >/dev/null 2>&1; then
    echo "Stopping ollama"
    kill "$OLLAMA_PID" 2>/dev/null || true
    wait "$OLLAMA_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

# Wait for ollama
for _ in {1..60}; do
  if curl --silent --fail "http://127.0.0.1:11434/api/tags" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

echo "====================================================================="
echo "Testing llama3.1:8b with MAX_CHARS=$MAX_CHARS"
echo "====================================================================="

python3 << 'PYTEST'
import sys
import time
from pathlib import Path
import os

sys.path.insert(0, "/prj/LINDA_LLM/scripts")

# Set args before importing documents
import argparse
sys.argv = ["test", "--model", "llama31", "--node", "local", "--data", "cardio", "--target", "ppi", "--doclevel", "docs"]

from documents import get_texts
from clients import ClientRegistry, ollama_client_names, ip_dict
from baml_client import b
from baml_client.types import PromptMessage
from prompts import prompts
import os

MAX_CHARS = int(os.environ.get("MAX_CHARS", "20000"))

# Setup client for llama31
cr = ClientRegistry()
for name, client_model in ollama_client_names:
    if name == "llama31":
        cr.add_llm_client(
            name=name,
            provider="openai-generic",
            options={
                "base_url": "http://127.0.0.1:11434/v1",
                "model": client_model,
                "max_tokens": 10000,
                "temperature": 0.0,
                "n_ctx": 120000,
            },
        )
        break
cr.set_primary("llama31")

# Get first 3 documents
print("Loading documents...")
docs, keys, names = get_texts()
test_docs = docs[:3]

print(f"Processing 3 documents with MAX_CHARS={MAX_CHARS}")
total_time = 0

for i, doc in enumerate(test_docs):
    text = doc[0].page_content
    original_len = len(text)
    
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS]
        print(f"Doc {i}: Truncated from {original_len} to {MAX_CHARS} chars")
    else:
        print(f"Doc {i}: Using {len(text)} chars")
    
    messages = []
    for prompt_text in prompts:
        messages.append(PromptMessage(role="system", content=prompt_text))
    messages.append(PromptMessage(role="user", content=text))
    
    start = time.time()
    try:
        response = b.GeneralChatExtractRelationships(messages, {"client": cr.get_primary()})
        elapsed = time.time() - start
        total_time += elapsed
        print(f"  Completed in {elapsed:.2f} seconds")
    except Exception as e:
        print(f"  ERROR: {e}")

avg = total_time / 3
print(f"\n{'='*60}")
print(f"MAX_CHARS={MAX_CHARS}")
print(f"Average time: {avg:.2f} seconds/doc")
print(f"Total time: {total_time:.2f} seconds for 3 docs")
print(f"{'='*60}")
PYTEST

echo ""
echo "Test complete. Check $OLLAMA_LOG for ollama details."
