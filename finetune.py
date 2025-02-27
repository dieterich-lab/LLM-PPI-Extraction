import json
import os

from baml.baml_client.type_builder import TypeBuilder
from baml.baml_client.types import Triple, Triples
from datasets import Dataset, load_dataset
from documents import docs
from huggingface_hub import login
from paths import regulatome_eval_path
from prompts import prompts, system_prompt
from pydantic.json import pydantic_encoder

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
            {"role": "system", "content": system_prompt},
            {"role": "system", "content": f"TEXT: {doc}"},
            {"role": "system", "content": f"{OUTPUT_FORMAT}"},
            {"role": "user", "content": prompts[0]},
            {"role": "assistant", "content": formatted_triples},
        ]
    }


with open(regulatome_eval_path, "r") as f:
    eval_data = [(x.split("\t")[0], x.split("\t")[1]) for x in f.readlines()[1:]]

eval_data = [{"file_stem": x[0], "relations": x[1]} for x in eval_data]


dataset = Dataset.from_list(eval_data)

# Transform to conversational format
dataset = dataset.map(chat_conversion, batched=False)
