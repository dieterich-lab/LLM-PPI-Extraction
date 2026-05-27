#!/usr/bin/env python3
"""Local HuggingFace extraction path for finetuned RegulaTome models.

This bypasses Ollama/BAML and runs generation directly from local model weights.
It is intended as a pragmatic fallback when Ollama model artifacts are unavailable.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ALIAS_TO_PATH = {
    "llama31regu": "/prj/LINDA_LLM/outputs/finetunedmodels/unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit_regulatome_ppi_merged_16bit",
    "llama31regutf": "/prj/LINDA_LLM/outputs/finetunedmodels/unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit_regulatome_tf_merged_16bit",
    "llama33regu": "/prj/LINDA_LLM/outputs/finetunedmodels/unsloth/Llama-3.3-70B-Instruct-bnb-4bit_regulatome",
    "llama33regutf": "/prj/LINDA_LLM/outputs/finetunedmodels/unsloth/Llama-3.3-70B-Instruct-bnb-4bit_regulatome_tf_merged_16bit",
}


SYSTEM_PROMPT = (
    "You are an expert molecular biologist. "
    "Extract only direct protein-protein interactions from the provided text. "
    "Return strict JSON only with this schema: "
    '{"triples":[{"head":"...","relation":"INTERACTS_WITH","tail":"...","confidence":"high|low"}]}. '
    'If none are present, return {"triples":[]}.'
)

USER_PROMPT_TEMPLATE = (
    "TASK: Extract direct protein-protein interactions only. "
    "Exclude co-expression, co-localization, and indirect regulation.\n\n"
    "TEXT:\n{text}\n"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", default="llama31regu", choices=list(MODEL_ALIAS_TO_PATH.keys())
    )
    parser.add_argument(
        "--input-dir", required=True, help="Directory containing .md/.txt papers"
    )
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--max-chars", type=int, default=20000)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    return parser.parse_args()


def build_messages(text: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(text=text)},
    ]


def find_json_object(raw: str) -> Dict:
    # Extract first JSON object in output. Keep this tolerant for model verbosity.
    match = re.search(r"\{[\s\S]*\}", raw)
    if not match:
        return {"triples": []}
    candidate = match.group(0)
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return {"triples": []}
    if not isinstance(parsed, dict):
        return {"triples": []}
    triples = parsed.get("triples", [])
    if not isinstance(triples, list):
        triples = []

    cleaned = []
    for triple in triples:
        if not isinstance(triple, dict):
            continue
        head = str(triple.get("head", "")).strip()
        relation = str(triple.get("relation", "")).strip() or "INTERACTS_WITH"
        tail = str(triple.get("tail", "")).strip()
        confidence = str(triple.get("confidence", "low")).strip().lower()
        if relation != "INTERACTS_WITH":
            relation = "INTERACTS_WITH"
        if confidence not in {"high", "low"}:
            confidence = "low"
        if head and tail:
            cleaned.append(
                {
                    "head": head,
                    "relation": relation,
                    "tail": tail,
                    "confidence": confidence,
                }
            )
    return {"triples": cleaned}


def main() -> None:
    args = parse_args()

    model_path = Path(MODEL_ALIAS_TO_PATH[args.model])
    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")

    input_dir = Path(args.input_dir)
    files = sorted(list(input_dir.glob("*.md")) + list(input_dir.glob("*.txt")))
    files = files[args.start : args.start + args.limit]

    if not files:
        print("No input files found for selected range.")
        return

    print(f"Loading model: {args.model} -> {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    model.eval()

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w") as out_f:
        for idx, file_path in enumerate(files):
            text = file_path.read_text(errors="ignore")
            if len(text) > args.max_chars:
                text = text[: args.max_chars]

            messages = build_messages(text)
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                return_token_type_ids=False,
            ).to(model.device)
            # Some llama checkpoints still surface token_type_ids; generation rejects them.
            inputs.pop("token_type_ids", None)

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                )

            generated_ids = output_ids[0][inputs["input_ids"].shape[1] :]
            raw_output = tokenizer.decode(generated_ids, skip_special_tokens=True)
            parsed = find_json_object(raw_output)

            row = {
                "filename": str(file_path),
                "responses": [parsed.get("triples", [])],
                "text": text,
                "raw_output": raw_output,
            }
            out_f.write(json.dumps(row) + "\n")
            out_f.flush()

            print(
                f"[{idx+1}/{len(files)}] {file_path.name}: triples={len(parsed.get('triples', []))}"
            )

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
