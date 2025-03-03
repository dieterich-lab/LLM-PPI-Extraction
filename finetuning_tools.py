import json

import torch
from baml.baml_client.types import Triple, Triples
from clients import hf_model_id
from datasets import Dataset, load_from_disk
from documents import docs
from paths import finetune_data_path, regulatome_eval_path
from peft import LoraConfig, prepare_model_for_kbit_training
from prompts import OUTPUT_FORMAT, prompts, rel_system_prompt
from pydantic.json import pydantic_encoder
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(
        hf_model_id, use_fast=True, trust_remote_code=True
    )

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    tokenizer.model_max_length = 120_000
    if hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None:
        tokenizer.chat_template = None  # Reset the chat template
    return tokenizer


def get_bnb_config():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    return bnb_config


def get_model(bnb_config):
    device_map = (
        {"": torch.cuda.current_device()} if torch.cuda.is_available() else None
    )
    model = AutoModelForCausalLM.from_pretrained(
        hf_model_id,
        device_map=device_map,
        attn_implementation="flash_attention_2",
        quantization_config=bnb_config,
    )
    return model


def get_peft_config():
    peft_config = LoraConfig(
        lora_alpha=128,  # 32
        lora_dropout=0.05,
        r=256,  # 16
        bias="none",
        target_modules=[
            "q_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "k_proj",
            "v_proj",
        ],
        task_type="CAUSAL_LM",
    )
    return peft_config


def get_dataset():
    if not finetune_data_path.exists():

        def chat_conversion(data):
            doc = [
                x for x in docs if x[0].metadata["file_path"].stem == data["file_stem"]
            ][0][0].page_content
            relations = [x.strip() for x in data["relations"].split(";")]
            relations = [(x.split("=")[0], x.split("=")[1]) for x in relations]
            triples = [
                Triple(head=x[0], relation="INTERACTS_WITH", tail=x[1])
                for x in relations
            ]

            triples = Triples(triples=triples)
            formatted_triples = f"```json\n{json.dumps(triples.model_dump(), default=pydantic_encoder, indent=2)}\n```"

            return {
                "messages": [
                    {"role": "system", "content": rel_system_prompt},
                    {"role": "system", "content": f"TEXT: {doc}"},
                    {
                        "role": "system",
                        "content": f"Use the following OUTPUT FORMAT:\n{OUTPUT_FORMAT}",
                    },
                    {"role": "user", "content": prompts[0]},
                    {"role": "assistant", "content": formatted_triples},
                ]
            }

        with open(regulatome_eval_path, "r") as f:
            eval_data = [
                (x.split("\t")[0], x.split("\t")[1]) for x in f.readlines()[1:]
            ]

        eval_data = [{"file_stem": x[0], "relations": x[1]} for x in eval_data]
        dataset = Dataset.from_list(eval_data)
        dataset = dataset.map(chat_conversion, batched=False)
        dataset.save_to_disk(finetune_data_path)

    dataset = load_from_disk(finetune_data_path)
    return dataset
