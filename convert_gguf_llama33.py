"""Re-run GGUF conversion from already-trained LoRA model."""

import os
import sys

import torch
from unsloth import FastLanguageModel

sys.path.append(str(__import__("pathlib").Path(__file__).resolve().parents[1]))

# Path to already-trained LoRA model
SFT_PATH = "/beegfs/prj/LINDA_LLM/outputs/finetunedmodels/unsloth/Llama-3.3-70B-Instruct-bnb-4bit_regulatome_ppi_lora"
GGUF_PATH = "/beegfs/prj/LINDA_LLM/outputs/finetunedmodels/unsloth/Llama-3.3-70B-Instruct-bnb-4bit_regulatome_ppi_GGUF"

print(f"Loading model from: {SFT_PATH}")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=SFT_PATH,
    max_seq_length=120_000,
    dtype=None,
    load_in_4bit=True,
)
print("Model loaded. Starting GGUF conversion...")
model.save_pretrained_gguf(
    GGUF_PATH,
    tokenizer,
    quantization_method=["q4_k_m"],
)
print(f"GGUF conversion complete. Saved to: {GGUF_PATH}")
