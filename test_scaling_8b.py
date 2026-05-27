#!/usr/bin/env python3
"""Test scaling behavior of llama3.1:8b with different context sizes."""

import time
import sys
from pathlib import Path

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent))

from documents import get_texts
from parser import args
from clients import cr
from baml_client import b
from baml_client.types import PromptMessage
from prompts import prompts

# Override args for this test
args.model = "llama31"
args.node = "local"
args.data = "cardio"

# Reinitialize client with llama31
from clients import ClientRegistry, ollama_client_names, ip_dict

cr = ClientRegistry()
port_dict = {"local": 34}
port = port_dict["local"]

for name, client_model in ollama_client_names:
    if name == "llama31":
        cr.add_llm_client(
            name=name,
            provider="openai-generic",
            options={
                "base_url": f"http://127.0.0.1:11434/v1",
                "model": client_model,
                "max_tokens": 10000,
                "temperature": 0.0,
                "n_ctx": 120_000,
            },
        )
cr.set_primary("llama31")

print("Testing llama3.1:8b scaling with different context sizes")
print("=" * 70)

# Get first 3 documents
docs, keys, names = get_texts()
test_docs = docs[:3]
test_keys = keys[:3]

for max_chars in [20000, 60000]:
    print(f"\n\nTesting with MAX_CHARS = {max_chars}")
    print("-" * 70)

    total_time = 0

    for i, (doc, key) in enumerate(zip(test_docs, test_keys)):
        text = doc[0].page_content

        # Truncate
        if len(text) > max_chars:
            print(f"Doc {i}: Truncating from {len(text)} to {max_chars} chars")
            text = text[:max_chars]
        else:
            print(f"Doc {i}: Using full text ({len(text)} chars)")

        # Build messages
        messages = []
        for prompt_text in prompts:
            messages.append(PromptMessage(role="system", content=prompt_text))
        messages.append(PromptMessage(role="user", content=text))

        # Time the extraction
        start = time.time()
        try:
            response = b.GeneralChatExtractRelationships(
                messages, {"client": cr.get_primary()}
            )
            elapsed = time.time() - start
            total_time += elapsed
            print(f"  Completed in {elapsed:.1f} seconds")
        except Exception as e:
            print(f"  ERROR: {e}")
            elapsed = 0

    avg_time = total_time / len(test_docs)
    print(f"\nAverage time per document: {avg_time:.1f} seconds")
    print(f"Total time for 3 documents: {total_time:.1f} seconds")

print("\n" + "=" * 70)
print("Test complete")
