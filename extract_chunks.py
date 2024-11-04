import argparse
import json
import os
import pickle
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_ollama import ChatOllama
from structured_classes import Sentences
from templates import PPI_EXTRACTION_SYSTEM
from utils import Timeout

parser = argparse.ArgumentParser()
parser.add_argument(
    "--task",
    nargs="?",
    const="tf",
    type=str,
    default="tf",
    choices=["tf", "ppi"],
)
parser.add_argument(
    "--parser",
    nargs="?",
    const="llama_parse",
    type=str,
    default="llama_parse",
    choices=["llama_parse", "marker"],
)
parser.add_argument(
    "--curated",
    action="store_true",
)
parser.add_argument(
    "--dev",
    action="store_true",
)
parser.add_argument("--model", choices=["70b"], default="70b")
args = parser.parse_args()

old = True

ending_dict = {"marker": "md", "llama_parse": "txt"}

model_dict = {
    "70b": "llama3.1:70b",
}
model = model_dict[args.model]
g4 = "10.250.135.153"
g2 = "10.250.135.143"
g3 = "10.250.135.150"
g5 = "10.250.135.156"
sg500 = "10.250.135.128"
port34 = 11434
port35 = 11435
port36 = 11436
llm = ChatOllama(
    model=model,
    temperature=0,
    keep_alive="24h",
    base_url=f"http://{g4}:{port36}",
    # num_ctx=12_800,
)

system_extraction_dict = {"ppi": PPI_EXTRACTION_SYSTEM, "tf": ""}
system_extraction_template = system_extraction_dict[args.task]

aper_dict = dict()

if old:
    _task = "ppi"
else:
    _task = args.task

_paper_paths = Path(
    f"/beegfs/prj/LINDA_LLM/outputs/parsed_papers/{_task}/{args.parser}"
)
if args.curated:
    _paper_paths = _paper_paths / "5curated"
else:
    _paper_paths = _paper_paths / "100samples"

paper_paths = list(_paper_paths.glob(f"*.{ending_dict[args.parser]}"))

paper_pkl_path = Path(
    f"/beegfs/prj/LINDA_LLM/outputs/paper_chunks/{_task}/{args.parser}/paper_chunks.pkl"
)
if args.curated:
    paper_pkl_path = paper_pkl_path.parent / "5curated" / "paper_chunks.pkl"
else:
    paper_pkl_path = paper_pkl_path.parent / "100samples" / "paper_chunks.pkl"

paper_dict_path = Path(
    f"/beegfs/prj/LINDA_LLM/outputs/paper_dicts/{_task}/{args.parser}/paper_dict.pkl"
)
if args.curated:
    paper_dict_path = paper_dict_path.parent / "5curated" / "paper_chunks.pkl"
else:
    paper_dict_path = paper_dict_path.parent / "100samples" / "paper_chunks.pkl"


if old:
    _task = "tf"
    chunk_filename = "chunks_documents_old.pkl"
else:
    chunk_filename = "chunks.pkl"
    _task = args.task

chunk_pkl_path = (
    f"/beegfs/prj/LINDA_LLM/outputs/extracted_chunks/{_task}/{args.parser}/{model}"
)

if args.curated:
    chunk_pkl_path = Path(chunk_pkl_path).parent / "5curated" / chunk_filename
else:
    chunk_pkl_path = Path(chunk_pkl_path).parent / "100samples" / chunk_filename

os.makedirs(Path(chunk_pkl_path).parent, exist_ok=True)

extraction_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            system_extraction_template,
        ),
        (
            "human",
            "Use the given format to extract the sentences. Here's the paper: {input}",
        ),
    ]
)
extraction_schema = convert_to_openai_tool(Sentences)
extraction_llm = llm.with_structured_output(extraction_schema, include_raw=True)
extraction_chain = extraction_prompt | extraction_llm

if not args.dev:
    f = open(chunk_pkl_path, "wb")

for i, x in enumerate(paper_paths):
    print(i, x)
    text = open(x, "r").read().strip()
    c = 0
    try:
        # while c < 5:
        # try:
        #     with Timeout(60):
        #         chunks = extraction_chain.invoke({"input": text})
        #         break
        # except Timeout.Timeout:
        #     print("Timeout")
        #     c += 1
        chunks = extraction_chain.invoke({"input": text})
        pickle.dump(chunks, f)
    except Exception as e:
        print(e)

if not args.dev:
    f.close()

print(f"Finished writing chunks to {chunk_pkl_path}.")
