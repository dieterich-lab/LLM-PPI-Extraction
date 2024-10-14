import argparse
import json
import os
import pickle
from pathlib import Path

from langchain_community.llms import Ollama
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from prompt_utils import create_unstructured_prompt
from templates import (
    PPI_BASESTRINGPARTS,
    PPI_EXAMPLES,
    PPI_INTERACTIONS,
    PPI_NODE_LABELS,
    TEMPLATE,
    TEMPLATE_SIMPLE,
    TF_BASESTRINGPARTS,
    TF_BASESTRINGPARTS_SIMPLE,
    TF_EXAMPLES,
    TF_EXAMPLES_SIMPLE,
    TF_INTERACTIONS,
    TF_NODE_LABELS,
)

# SAMPLES = True

parser = argparse.ArgumentParser()
parser.add_argument(
    "--parser",
    nargs="?",
    const="llama_parse",
    type=str,
    default="llama_parse",
    choices=["llama_parse", "marker"],
)
parser.add_argument(
    "--simple",
    action="store_true",
)
parser.add_argument(
    "--curated",
    action="store_true",
)
parser.add_argument("--model", choices=["8b", "70b"], default="8b")
args = parser.parse_args()


# task = "ppi"
task = "tf"
old = True


ending_dict = {"marker": "md", "llama_parse": "txt"}
basestring_dict = {"ppi": PPI_BASESTRINGPARTS, "tf": TF_BASESTRINGPARTS}
example_dict = {"ppi": PPI_EXAMPLES, "tf": TF_EXAMPLES}
interactions_dict = {"ppi": PPI_INTERACTIONS, "tf": TF_INTERACTIONS}
nodelabels_dict = {"ppi": PPI_NODE_LABELS, "tf": TF_NODE_LABELS}


model_dict = {"8b": "llama3.1:8b", "70b": "llama3.1:70b"}
model = model_dict[args.model]
g4 = "10.250.135.153"
g2 = "10.250.135.143"
g3 = "10.250.135.150"
g5 = "10.250.135.156"
port34 = 11434
port35 = 11435
port36 = 11436
llm = Ollama(
    model=model, base_url=f"http://{g4}:{port35}", temperature=0, keep_alive="24h"
)


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    length_function=len,
    is_separator_regex=False,
)
paper_dict = dict()

if old:
    _task = "ppi"
else:
    _task = task

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
    f"/beegfs/prj/LINDA_LLM/outpts/paper_dicts/_{task}/{args.parser}/paper_dict.pkl"
)
if args.curated:
    paper_dict_path = paper_dict_path.parent / "5curated" / "paper_chunks.pkl"
else:
    paper_dict_path = paper_dict_path.parent / "100samples" / "paper_chunks.pkl"

os.makedirs(Path(paper_pkl_path).parent, exist_ok=True)
f = open(paper_pkl_path, "wb")
for i, x in enumerate(paper_paths):
    print(i, x)
    paper_dict[i] = str(x)
    text = open(x, "r").read().strip()
    if text:
        texts = text_splitter.create_documents([text])
        for t in texts:
            pickle.dump((t, i), f)
f.close()
os.makedirs(Path(paper_dict_path).parent, exist_ok=True)
with open(paper_dict_path, "w") as f:
    json.dump(paper_dict, f, indent=4)

with open(paper_dict_path, "r") as f:
    paper_dict = json.load(f)
    paper_dict = {int(k): v for k, v in paper_dict.items()}

documents = list()
with open(paper_pkl_path, "rb") as f:
    while 1:
        try:
            documents.append(pickle.load(f))
        except EOFError:
            break
print(len(documents))

prompt = create_unstructured_prompt(
    base_string_parts=(
        basestring_dict[task] if not args.simple else TF_BASESTRINGPARTS_SIMPLE
    ),
    template=TEMPLATE if not args.simple else TEMPLATE_SIMPLE,
    examples=example_dict[task] if not args.simple else TF_EXAMPLES_SIMPLE,
    node_labels=False if args.simple else nodelabels_dict[task],
    rel_types=False if args.simple else interactions_dict[task],
)

llm_transformer = LLMGraphTransformer(
    llm=llm,
    allowed_nodes=[] if args.simple else nodelabels_dict[task],
    allowed_relationships=[] if args.simple else interactions_dict[task],
    strict_mode=not args.simple,
    prompt=prompt,
)

if old:
    _task = "tf"
    graph_doc_filename = "graph_documents_old.pkl"
else:
    graph_doc_filename = "graph_documents.pkl"
    _task = task

graphdoc_pkl_path = f"/beegfs/prj/LINDA_LLM/outputs/graph_docs/{_task}/{args.parser}/{model}/graph_documents.pkl"

if args.curated:
    graphdoc_pkl_path = Path(graphdoc_pkl_path).parent / "5curated" / graph_doc_filename
else:
    graphdoc_pkl_path = (
        Path(graphdoc_pkl_path).parent / "100samples" / graph_doc_filename
    )

if args.simple:
    graphdoc_pkl_path = Path(graphdoc_pkl_path).parent / "simple" / graph_doc_filename

os.makedirs(Path(graphdoc_pkl_path).parent, exist_ok=True)

graph_documents = list()
f = open(graphdoc_pkl_path, "wb")
for i, (doc, id) in enumerate(documents):
    print(i, id)
    try:
        graph_doc = llm_transformer.convert_to_graph_documents([doc])[0]
        graph_doc.source.metadata["source"] = paper_dict[id]
        graph_doc.source.metadata["id"] = str(id)
        pickle.dump(graph_doc, f)
    except Exception as e:
        print(e)
f.close()

print(f"Finished writing graph docs to {graphdoc_pkl_path}.")
