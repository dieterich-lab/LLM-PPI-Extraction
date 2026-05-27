#!/usr/bin/env python3
"""Local HuggingFace extraction path for finetuned RegulaTome models.

This bypasses Ollama/BAML and runs generation directly from local model weights.
It is intended as a pragmatic fallback when Ollama model artifacts are unavailable.
"""

import argparse
import importlib
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

MODEL_ALIAS_TO_PATH = {
    "llama31regu": "/prj/LINDA_LLM/outputs/finetunedmodels/unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit_regulatome_ppi_merged_16bit",
    "llama31regutf": "/prj/LINDA_LLM/outputs/finetunedmodels/unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit_regulatome_tf_merged_16bit",
    "llama33regu": "/prj/LINDA_LLM/outputs/finetunedmodels/unsloth/Llama-3.3-70B-Instruct-bnb-4bit_regulatome",
    "llama33regutf": "/prj/LINDA_LLM/outputs/finetunedmodels/unsloth/Llama-3.3-70B-Instruct-bnb-4bit_regulatome_tf_merged_16bit",
}


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
    parser.add_argument("--target", choices=["ppi", "tf", "ppitf"], default="ppi")
    parser.add_argument(
        "--extractionmode", choices=["direct", "nerrel"], default="direct"
    )
    parser.add_argument(
        "--chattype", choices=["oneshot", "stepwise"], default="oneshot"
    )
    parser.add_argument(
        "--data",
        choices=[
            "cardio",
            "cardiac",
            "regulatome",
            "5curated",
            "eval",
            "biored",
            "regulatomepapers",
        ],
        default="cardio",
    )
    parser.add_argument("--noconfidence", action="store_true")
    parser.add_argument("--force-cot", action="store_true")
    parser.add_argument("--recall", action="store_true")
    parser.add_argument("--examples", choices=["neg", "pos", "negpos"])
    parser.add_argument("--all-nes-given", action="store_true")
    parser.add_argument("--true-nes-given", action="store_true")
    parser.add_argument("--spacy-nes-given", action="store_true")
    parser.add_argument("--lookup", action="store_true")
    parser.add_argument("--dynex-k", type=int, default=0)
    return parser.parse_args()


def load_ollama_prompts(args: argparse.Namespace) -> Tuple[str, List[str]]:
    """Load rel_system_prompt and prompts from prompts.py using parser-compatible args."""
    original_argv = sys.argv[:]
    parser_argv = [
        "hf_prompt_bootstrap",
        "--model",
        args.model,
        "--target",
        args.target,
        "--extractionmode",
        args.extractionmode,
        "--chattype",
        args.chattype,
        "--data",
        args.data,
    ]

    if args.noconfidence:
        parser_argv.append("--noconfidence")
    if args.force_cot:
        parser_argv.append("--force_cot")
    if args.recall:
        parser_argv.append("--recall")
    if args.examples:
        parser_argv.extend(["--examples", args.examples])
    if args.all_nes_given:
        parser_argv.append("--all_nes_given")
    if args.true_nes_given:
        parser_argv.append("--true_nes_given")
    if args.spacy_nes_given:
        parser_argv.append("--spacy_nes_given")
    if args.lookup:
        parser_argv.append("--lookup")
    if args.dynex_k > 0:
        parser_argv.extend(["--dynex_k", str(args.dynex_k)])

    for module_name in ["prompts", "parser"]:
        if module_name in sys.modules:
            del sys.modules[module_name]

    try:
        sys.argv = parser_argv
        prompts_module = importlib.import_module("prompts")
        rel_system_prompt = str(prompts_module.rel_system_prompt)
        prompt_list = list(prompts_module.prompts)
    finally:
        sys.argv = original_argv

    return rel_system_prompt, prompt_list


def build_messages(
    text: str,
    rel_system_prompt: str,
    prompt_list: List[str],
    extraction_mode: str,
    spacy_nes_given: bool,
) -> List[Dict[str, str]]:
    """Mirror the oneshot/direct and nerrel prompt/message sequencing from extract.py."""
    prompts_copy = prompt_list.copy()

    messages: List[Dict[str, str]] = []
    messages.append({"role": "assistant", "content": rel_system_prompt})

    if not spacy_nes_given and prompts_copy:
        first_prompt = prompts_copy.pop(0)
        messages.append({"role": "user", "content": first_prompt})

    messages.append(
        {
            "role": "user",
            "content": (
                "Return ONLY strict JSON with this exact schema and no markdown/no extra keys: "
                '{"triples":[{"head":"...","relation":"INTERACTS_WITH","tail":"...","confidence":"high|low"}]}. '
                'If none exist, return {"triples":[]}. '
                "Do not include explanations or copied source text."
            ),
        }
    )
    messages.append({"role": "user", "content": f"\n\nTEXT: {text}"})

    if extraction_mode == "nerrel" and prompts_copy:
        first_rel_prompt = prompts_copy.pop(0)
        messages.append({"role": "user", "content": first_rel_prompt})

    return messages


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
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    args = parse_args()
    rel_system_prompt, prompt_list = load_ollama_prompts(args)

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
    print(
        f"Prompt source: prompts.py ({args.target}/{args.extractionmode}/{args.chattype})"
    )
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
            if args.max_chars > 0 and len(text) > args.max_chars:
                text = text[: args.max_chars]

            messages = build_messages(
                text,
                rel_system_prompt,
                prompt_list,
                args.extractionmode,
                args.spacy_nes_given,
            )
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
