#!/usr/bin/env python3
"""Standalone document cache builder.

Imports documents.py which triggers cache building at module level.
Run this once before launching sharded extraction jobs to avoid
concurrent cache writes.

Usage:
    python build_cache.py --data cardiac --target ppi --model llama33 [--force_new]

All arguments from extract.py's parser are accepted. The script simply
loads the document cache and exits.
"""

import sys
import time

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))


def main():
    t0 = time.monotonic()
    print("Building document cache...")
    import documents  # noqa: F401  — triggers get_texts() → write_documents()

    elapsed = time.monotonic() - t0
    print(f"Cache ready ({len(documents.texts)} documents loaded in {elapsed:.1f}s).")


if __name__ == "__main__":
    main()
