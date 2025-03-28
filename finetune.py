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
from finetuning_tools import get_dataset
from huggingface_hub import login
from transformers import TrainingArguments
from trl import SFTTrainer

hf_key = os.getenv("HF_ACCESS_TOKEN")
login(token=hf_key, add_to_git_credential=True)

max_seq_length = 120_000  # Choose any! We auto support RoPE Scaling internally!
dtype = None
load_in_4bit = True  # Use 4bit quantization to reduce memory usage. Can be False.


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=hf_model_id,
    max_seq_length=max_seq_length,
    dtype=dtype,  # auto detection
    load_in_4bit=load_in_4bit,
    token=hf_key,
)

tokenizer = get_chat_template(
    tokenizer,
    chat_template="chatml",  # Supports zephyr, chatml, mistral, llama, alpaca, vicuna, vicuna_old, unsloth
    map_eos_token=True,  # Maps <|im_end|> to </s> instead
)

train_dataset, dev_dataset, test_dataset = get_dataset(tokenizer, force_new=True)
# train_dataset, dev_dataset, test_dataset = get_dataset(tokenizer)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0,  # Supports any, but = 0 is optimized
    bias="none",  # Supports any, but = "none" is optimized
    use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
    random_state=3407,
    use_rslora=False,  # We support rank stabilized LoRA
    loftq_config=None,  # And LoftQ
)

model.config.text_config.use_cache = False


trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,  # Can make training 5x faster for short sequences.
    args=TrainingArguments(
        disable_tqdm=True,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        # eval_strategy="epoch",
        do_eval=False,
        num_train_epochs=2,
        # max_steps=10,
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir=sft_model_path,
        report_to="none",  # Use this for WandB etc
    ),
)
trainer.model.config.use_cache = False

if False:
    trainer_stats = trainer.train()

# Loading
if False:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(sft_model_path),
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )
    FastLanguageModel.for_inference(model)  # Enable native 2x faster inference

# Saving
if True:
    model.save_pretrained_gguf(
        f"{sft_model_path}.GGUF", tokenizer, quantization_method=["q4_k_m", "q8_0"]
    )
    model.save_pretrained_gguf(
        f"{sft_model_path}.GGUF",
        tokenizer,
        quantization_type="q4_k_m",
    )

# Pushing
if False:
    model.push_to_hub_gguf(
        f"phiwi/{Path(hf_model_id).name}-regulatome", tokenizer, token=hf_key
    )


if True:
    inputs = tokenizer(test_dataset[0]["text"], return_tensors="pt").to("cuda")

    # outputs = model.generate(**inputs, max_new_tokens=5_000, use_cache=True)
    # print(tokenizer.batch_decode(outputs))

    text_streamer = TextStreamer(tokenizer)
    _ = model.generate(**inputs, streamer=text_streamer, max_new_tokens=5_000)
    pass
