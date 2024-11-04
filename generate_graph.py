import json
import os
import pickle
from parser import args
from pathlib import Path

from langchain_community.llms import Ollama
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_ollama.llms import OllamaLLM
from langchain_text_splitters import MarkdownTextSplitter
from prompt_utils import create_unstructured_prompt
from structured_classes import (
    PPI_Triples,
    PPI_Triples_Simple,
    TF_Triples,
    TF_Triples_Simple,
)
from templates import (
    PPI_BASESTRINGPARTS,
    PPI_BASESTRINGPARTS_SIMPLE,
    PPI_EXAMPLES,
    PPI_EXAMPLES_SIMPLE,
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
    TF_TEMPLATE,
    TF_TEMPLATE_SIMPLE,
)
from utils import Timeout

old = True

ending_dict = {"marker": "md", "llama_parse": "txt"}
basestring_dict = {"ppi": PPI_BASESTRINGPARTS, "tf": TF_BASESTRINGPARTS}
simple_basestring_dict = {
    "ppi": PPI_BASESTRINGPARTS_SIMPLE,
    "tf": TF_BASESTRINGPARTS_SIMPLE,
}
example_dict = {"ppi": PPI_EXAMPLES, "tf": TF_EXAMPLES}
simple_example_dict = {"ppi": PPI_EXAMPLES_SIMPLE, "tf": TF_EXAMPLES_SIMPLE}
interactions_dict = {"ppi": PPI_INTERACTIONS, "tf": TF_INTERACTIONS}
nodelabels_dict = {"ppi": PPI_NODE_LABELS, "tf": TF_NODE_LABELS}


model_dict = {
    "8b": "llama3.1:8b",
    "70b": "llama3.1:70b",
    "405b": "llama3.1:405b",
    "mixtral": "mixtral:8x22b",
    "biollm": "taozhiyuai/openbiollm-llama-3:70b_q4_k_m",
}
model = model_dict[args.model]

ip_dict = {
    "g4": "10.250.135.153",
    "g2": "10.250.135.143",
    "g3": "10.250.135.150",
    "g5": "10.250.135.156",
}
# llm = OllamaLLM(

llm = Ollama(
    model=model,
    temperature=0,
    keep_alive="24h",
    base_url=f"http://{ip_dict[args.gpu]}:114{args.port}",
)

text_splitter = MarkdownTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    length_function=len,
    is_separator_regex=False,
)
paper_dict = dict()

if old and args.target in ["tf", "both"]:
    _task = "ppi"
else:
    _task = args.target

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

schema_dict = {"ppi": PPI_Triples, "tf": TF_Triples}
simple_schema_dict = {"ppi": PPI_Triples_Simple, "tf": TF_Triples_Simple}
if args.target != "both":
    schema = (
        schema_dict[args.target] if not args.simple else simple_schema_dict[args.target]
    )
    prompt = create_unstructured_prompt(
        base_string_parts=(
            basestring_dict[args.target]
            if not args.simple
            else simple_basestring_dict[args.target]
        ),
        template=TEMPLATE if not args.simple else TEMPLATE_SIMPLE,
        examples=(
            example_dict[args.target]
            if not args.simple
            else simple_example_dict[args.target]
        ),
        schema=schema,
        node_labels=False if args.simple else nodelabels_dict[args.target],
        rel_types=False if args.simple else interactions_dict[args.target],
    )
    llm_transformer = LLMGraphTransformer(
        llm=llm,
        allowed_nodes=[] if args.simple else nodelabels_dict[args.target],
        allowed_relationships=[] if args.simple else interactions_dict[args.target],
        strict_mode=not args.simple,
        prompt=prompt,
    )

else:
    schema = PPI_Triples if not args.simple else PPI_Triples_Simple
    ppi_prompt = create_unstructured_prompt(
        base_string_parts=(
            PPI_BASESTRINGPARTS if not args.simple else PPI_BASESTRINGPARTS_SIMPLE
        ),
        template=TEMPLATE if not args.simple else TEMPLATE_SIMPLE,
        examples=(PPI_EXAMPLES if not args.simple else PPI_EXAMPLES_SIMPLE),
        schema=schema,
        node_labels=False if args.simple else PPI_NODE_LABELS,
        rel_types=False if args.simple else PPI_INTERACTIONS,
    )
    ppi_llm_transformer = LLMGraphTransformer(
        llm=llm,
        allowed_nodes=[] if args.simple else PPI_NODE_LABELS,
        allowed_relationships=[] if args.simple else PPI_INTERACTIONS,
        strict_mode=not args.simple,
        prompt=ppi_prompt,
    )

if old and args.target == "tf":
    graph_doc_filename = "graph_documents_old.pkl"
else:
    graph_doc_filename = "graph_documents.pkl"

graphdoc_pkl_path = f"/beegfs/prj/LINDA_LLM/outputs/graph_docs/{args.target}/{args.parser}/{model}/graph_documents.pkl"

if args.curated:
    graphdoc_pkl_path = Path(graphdoc_pkl_path).parent / "5curated" / graph_doc_filename
else:
    graphdoc_pkl_path = (
        Path(graphdoc_pkl_path).parent / "100samples" / graph_doc_filename
    )

if args.simple:
    graphdoc_pkl_path = Path(graphdoc_pkl_path).parent / "simple" / graph_doc_filename

os.makedirs(Path(graphdoc_pkl_path).parent, exist_ok=True)

if args.target == "both":
    ppi_graphdoc_pkl_path = Path(graphdoc_pkl_path).parent / "ppi_graph_documents.pkl"
    tf_graphdoc_pkl_path = Path(graphdoc_pkl_path).parent / "tf_graph_documents.pkl"
    os.makedirs(Path(ppi_graphdoc_pkl_path).parent, exist_ok=True)
    os.makedirs(Path(tf_graphdoc_pkl_path).parent, exist_ok=True)

graph_documents = list()

if not args.dev:
    if args.target != "both":
        f = open(graphdoc_pkl_path, "wb")
    else:
        ppi_f = open(ppi_graphdoc_pkl_path, "wb")
        tf_f = open(tf_graphdoc_pkl_path, "wb")

for i, (doc, id) in enumerate(documents):
    print(i, id)
    c = 0
    try:
        if args.target != "both":
            graph_doc = llm_transformer.convert_to_graph_documents([doc])[0]
            graph_doc.source.metadata["source"] = paper_dict[id]
            graph_doc.source.metadata["id"] = str(id)
            if not args.dev:
                pickle.dump(graph_doc, f)
        else:
            while c < 5:
                try:
                    with Timeout(60):
                        ppi_graph_doc = ppi_llm_transformer.convert_to_graph_documents(
                            [doc]
                        )[0]
                        break
                except Timeout.Timeout:
                    print("Timeout")
                    c += 1
            # ppi_graph_doc = ppi_llm_transformer.convert_to_graph_documents([doc])[0]
            ppi_graph_doc.source.metadata["source"] = paper_dict[id]
            ppi_graph_doc.source.metadata["id"] = str(id)
            if not args.dev:
                pickle.dump(ppi_graph_doc, ppi_f)
            rel_triples = [
                [r.source.id, r.type, r.target.id] for r in ppi_graph_doc.relationships
            ]
            schema = TF_Triples if not args.simple else TF_Triples_Simple
            tf_prompt = create_unstructured_prompt(
                base_string_parts=(
                    TF_BASESTRINGPARTS if not args.simple else TF_BASESTRINGPARTS_SIMPLE
                ),
                template=TF_TEMPLATE if not args.simple else TF_TEMPLATE_SIMPLE,
                examples=(TF_EXAMPLES if not args.simple else TF_EXAMPLES_SIMPLE),
                schema=schema,
                node_labels=False if args.simple else PPI_NODE_LABELS,
                rel_types=False if args.simple else PPI_INTERACTIONS,
                previous_examples=rel_triples,
            )
            tf_llm_transformer = LLMGraphTransformer(
                llm=llm,
                allowed_nodes=[] if args.simple else TF_NODE_LABELS,
                allowed_relationships=([] if args.simple else TF_INTERACTIONS),
                strict_mode=not args.simple,
                prompt=tf_prompt,
            )
            while c < 5:
                try:
                    with Timeout(60):
                        tf_graph_doc = tf_llm_transformer.convert_to_graph_documents(
                            [doc]
                        )[0]
                        break
                except Timeout.Timeout:
                    print("Timeout")
                    c += 1
            # tf_graph_doc = tf_llm_transformer.convert_to_graph_documents([doc])[0]
            tf_graph_doc.source.metadata["source"] = paper_dict[id]
            tf_graph_doc.source.metadata["id"] = str(id)
            if not args.dev:
                pickle.dump(tf_graph_doc, tf_f)
    except Exception as e:
        print(e)

if not args.dev:
    if args.target != "both":
        f.close()
    else:
        ppi_f.close()
        tf_f.close()

print(f"Finished writing graph docs to {graphdoc_pkl_path}.")
