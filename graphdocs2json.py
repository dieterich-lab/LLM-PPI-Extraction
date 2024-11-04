import argparse
import json
import os
import pickle
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument(
    "--task",
    nargs="?",
    const="tf",
    type=str,
    default="tf",
    choices=["tf", "ppi", "both"],
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
    "--simple",
    action="store_true",
)
parser.add_argument(
    "--curated",
    action="store_true",
)
parser.add_argument(
    "--model", choices=["8b", "70b", "405b", "mixtral", "biollm"], default="8b"
)
args = parser.parse_args()

model_dict = {
    "8b": "llama3.1:8b",
    "70b": "llama3.1:70b",
    "405b": "llama3.1:405b",
    "mixtral": "mixtral:8x22b",
    "biollm": "taozhiyuai/openbiollm-llama-3:70b_q4_k_m",
}
model = model_dict[args.model]

old = True
# old = False

if old and args.task == "tf":
    graph_doc_filename = "graph_documents_old.pkl"
else:
    graph_doc_filename = "graph_documents.pkl"

graphdoc_pkl_path = (
    Path(f"/beegfs/prj/LINDA_LLM/outputs/graph_docs/{args.task}/{args.parser}/{model}")
    / graph_doc_filename
)

if args.curated:
    graphdoc_pkl_path = Path(graphdoc_pkl_path).parent / "5curated" / graph_doc_filename
else:
    graphdoc_pkl_path = (
        Path(graphdoc_pkl_path).parent / "100samples" / graph_doc_filename
    )

if args.simple:
    graphdoc_pkl_path = Path(graphdoc_pkl_path).parent / "simple" / graph_doc_filename

if args.task != "both":
    graphdoc_pkl_paths = [graphdoc_pkl_path]
else:
    ppi_graphdoc_pkl_path = Path(graphdoc_pkl_path).parent / "ppi_graph_documents.pkl"
    tf_graphdoc_pkl_path = Path(graphdoc_pkl_path).parent / "tf_graph_documents.pkl"
    graphdoc_pkl_paths = [ppi_graphdoc_pkl_path, tf_graphdoc_pkl_path]

for path in graphdoc_pkl_paths:
    graph_documents = list()
    # print(f"loading from {graphdoc_pkl_path}")

    with open(path, "rb") as f:
        while 1:
            try:
                graph_documents.append(pickle.load(f))
            except EOFError:
                break

    # print(len(graph_documents))

    rels = [
        (r, g.source.page_content) for g in graph_documents for r in g.relationships
    ]
    rel_triples = [[[r.source.id, r.type, r.target.id], p] for (r, p) in rels]

    if old and args.task == "tf":
        triple_filename = "triples_old.json"
    else:
        triple_filename = "triples.json"

    if args.task == "both":
        if "ppi" in path.stem:
            triple_filename = "ppi_triples.json"
        else:
            triple_filename = "tf_triples.json"

    triple_path = (
        Path(
            f"/beegfs/prj/LINDA_LLM/outputs/graph_triples/{args.task}/{args.parser}/{model}"
        )
        / triple_filename
    )

    if args.curated:
        triple_path = Path(triple_path).parent / "5curated" / triple_filename
    else:
        triple_path = Path(triple_path).parent / "100samples" / triple_filename

    if args.simple:
        triple_path = Path(triple_path).parent / "simple" / triple_filename

    os.makedirs(Path(triple_path).parent, exist_ok=True)

    with open(triple_path, "w") as f:
        json.dump(rel_triples, f, indent=4)

    print(f"{triple_path}")
