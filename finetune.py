import json
import os
from pathlib import Path

import torch
from baml.baml_client.types import Triple, Triples
from datasets import Dataset, load_from_disk
from documents import docs
from huggingface_hub import login
from paths import regulatome_eval_path
from peft import LoraConfig, prepare_model_for_kbit_training
from prompts import prompts, rel_system_prompt
from pydantic.json import pydantic_encoder
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer, setup_chat_format

from baml.baml_client.sync_client import b  # isort:skip

hf_key = os.getenv("HF_ACCESS_TOKEN")
login(token=hf_key, add_to_git_credential=True)


OUTPUT_FORMAT = """
Use the following OUTPUT FORMAT:{
    // list of triples that describe interactions between two biological entities
    triples: [
    {
        // head entity of the triple 
        head: string,
        // relationship type
        relation: "INTERACTS_WITH",
        // tail entity name of the triple
        tail: string,
    }
    ],
}
"""


def chat_conversion(data):
    doc = [x for x in docs if x[0].metadata["file_path"].stem == data["file_stem"]][0][
        0
    ].page_content
    relations = [x.strip() for x in data["relations"].split(";")]
    relations = [(x.split("=")[0], x.split("=")[1]) for x in relations]
    triples = [
        Triple(head=x[0], relation="INTERACTS_WITH", tail=x[1]) for x in relations
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
    eval_data = [(x.split("\t")[0], x.split("\t")[1]) for x in f.readlines()[1:]]

eval_data = [{"file_stem": x[0], "relations": x[1]} for x in eval_data]


# dataset = Dataset.from_list(eval_data)
# dataset = dataset.map(chat_conversion, batched=False)

data_path = Path("/prj/LINDA_LLM/outputs/datasets") / "regulatome.hf"
# dataset.save_to_disk(data_path)
dataset = load_from_disk(data_path)
# dataset = Dataset.load_from_disk(data_path)
model_id = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
# model_id = "meta-llama/Meta-Llama-3.1-8B"

tokenizer = AutoTokenizer.from_pretrained(
    model_id, use_fast=True, trust_remote_code=True
)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "left"

tokenizer.model_max_length = 120_000

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
device_map = {"": torch.cuda.current_device()} if torch.cuda.is_available() else None

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map=device_map,
    attn_implementation="flash_attention_2",
    quantization_config=bnb_config,
)

if hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None:
    tokenizer.chat_template = None  # Reset the chat template

model, tokenizer = setup_chat_format(model, tokenizer)
model = prepare_model_for_kbit_training(model)

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

sft_model_path = Path("/prj/LINDA_LLM/outputs") / "finetunedmodels" / model_id
args = TrainingArguments(
    output_dir=sft_model_path,  # directory to save the model and repository id
    num_train_epochs=2,  # number of training epochs
    per_device_train_batch_size=4,  # batch size per device during training
    gradient_accumulation_steps=2,  # number of steps before performing a backward/update pass
    gradient_checkpointing=True,  # use gradient checkpointing to save memory, use in distributed training
    optim="adamw_8bit",  # choose paged_adamw_8bit if not enough memory
    logging_steps=10,  # log every 10 steps
    save_strategy="epoch",  # save checkpoint every epoch
    learning_rate=2e-4,  # learning rate, based on QLoRA paper
    bf16=True,  # use bfloat16 precision
    tf32=True,  # use tf32 precision
    max_grad_norm=0.3,  # max gradient norm based on QLoRA paper
    warmup_ratio=0.03,  # warmup ratio based on QLoRA paper
    lr_scheduler_type="constant",  # use constant learning rate scheduler
    # push_to_hub=True,  # push model to Hugging Face hub
    # hub_model_id="llama3-8b-sft-qlora-re",
    report_to="tensorboard",  # report metrics to tensorboard
)

trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=dataset,
    peft_config=peft_config,
    max_seq_length=512,
    tokenizer=tokenizer,
    packing=False,  # True if the dataset is large
    dataset_kwargs={
        "add_special_tokens": False,  # the template adds the special tokens
        "append_concat_token": False,  # no need to add additional separator token
    },
)

trainer.train()
trainer.save_model()

pass
