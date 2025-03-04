import os
import sys
from pathlib import Path

from paths import sft_model_path

sys.path.append("..")  # isort:skip
from finetuning_tools import (
    get_bnb_config,
    get_dataset,
    get_model,
    get_peft_config,
    get_tokenizer,
)
from huggingface_hub import login
from peft import prepare_model_for_kbit_training
from transformers import TrainingArguments
from trl import SFTTrainer, setup_chat_format

hf_key = os.getenv("HF_ACCESS_TOKEN")
login(token=hf_key, add_to_git_credential=True)


tokenizer = get_tokenizer()
bnb_config = get_bnb_config()
peft_config = get_peft_config()
model = get_model(bnb_config)
dataset = get_dataset()

model, tokenizer = setup_chat_format(model, tokenizer)
model = prepare_model_for_kbit_training(model)

training_args = TrainingArguments(
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
    report_to="tensorboard",  # report metrics to tensorboard
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    peft_config=peft_config,
    max_seq_length=120_000,
    tokenizer=tokenizer,
    packing=False,  # True if the dataset is large
    dataset_kwargs={
        "add_special_tokens": False,  # the template adds the special tokens
        "append_concat_token": False,  # no need to add additional separator token
    },
)

trainer.train()
trainer.save_model()
