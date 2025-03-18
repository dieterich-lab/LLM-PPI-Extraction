import os
import sys
from pathlib import Path

from clients import hf_model_id

from unsloth import FastLanguageModel  # isort:skip
from unsloth import is_bfloat16_supported  # isort:skip
from unsloth.chat_templates import get_chat_template  # isort:skip

from paths import sft_model_path  # isort:skip
from transformers import TrainingArguments  # isort:skip
from transformers import TextStreamer  # isort:skip
from trl import SFTTrainer  # isort:skip

sys.path.append("..")  # isort:skip
import pickle
from parser import args

import outlines
from converter import convert_and_save_triples_to_json
from documents import all_ner_paths, texts
from finetuning_tools import get_dataset
from huggingface_hub import login
from paths import triple_json_path, triple_pkl_path
from prompts import OUTPUT_FORMAT, prompts, rel_system_prompt
from relations import Triples

hf_key = os.getenv("HF_ACCESS_TOKEN")
login(token=hf_key, add_to_git_credential=True)

max_seq_length = 120_000  # Choose any! We auto support RoPE Scaling internally!
dtype = None
load_in_4bit = False  # Use 4bit quantization to reduce memory usage. Can be False.

# Loading
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=str(sft_model_path),
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)
FastLanguageModel.for_inference(model)  # Enable native 2x faster inference

tokenizer = get_chat_template(
    tokenizer,
    chat_template="chatml",  # Supports zephyr, chatml, mistral, llama, alpaca, vicuna, vicuna_old, unsloth
    map_eos_token=True,  # Maps <|im_end|> to </s> instead
)

model = outlines.models.transformers(str(sft_model_path))
generator = outlines.generate.json(model, Triples)


def extract_rels(messages, responses, text, prompts):
    prompt_list = [
        {"role": "system", "content": rel_system_prompt},
        {"role": "system", "content": text},
        {
            "role": "system",
            "content": f"Use the following OUTPUT FORMAT:\n{OUTPUT_FORMAT}",
        },
        {"role": "user", "content": prompts[0]},
    ]
    prompt = tokenizer.apply_chat_template(
        prompt_list, tokenize=False, add_generation_prompt=True
    )
    response = generator(prompt)
    messages.append(response)
    return prompts


def main():
    mode = "wb" if not args.dev else "rb"
    with open(triple_pkl_path, mode) as triple_pkl_file:
        for i, doc in enumerate(texts):
            _prompts = prompts.copy()
            print(f"Doc {i}")
            text = doc[0].page_content
            messages = list()
            responses = list()
            if args.extractionmode == "nerrel" or args.all_ners_given:
                pass
            extract_rels(messages, responses, text, _prompts)
            if not args.dev:
                pickle.dump(
                    (responses, doc[0].page_content, doc[0].metadata["file_path"]),
                    triple_pkl_file,
                )

            if args.dev:
                break

    convert_and_save_triples_to_json(triple_pkl_path, triple_json_path)


if __name__ == "__main__":
    main()
