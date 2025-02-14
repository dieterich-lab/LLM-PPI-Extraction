import random

import transformers
from documents import docs
from tqdm import tqdm
from transformers import GPT2Tokenizer

transformers.utils.logging.set_verbosity_error()

print(len(docs))

tokens = 0
samples = random.sample(docs, 50)
for doc in tqdm(samples):
    text = doc[0].page_content
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    encoded_input = tokenizer(text)["input_ids"]
    tokens += len(encoded_input)

print(tokens / len(samples))
