import json
import os
import pickle
from parser import args
from pathlib import Path

from llm import model

graph_doc_filename = "graph_documents.pkl"

graphdoc_pkl_path = (
    Path(
        f"/beegfs/prj/LINDA_LLM/outputs/graph_docs/{args.target}/{args.parser}/{model}"
    )
    / graph_doc_filename
)

if args.curated:
    graphdoc_pkl_path = Path(graphdoc_pkl_path).parent / "5curated" / graph_doc_filename
else:
    graphdoc_pkl_path = (
        Path(graphdoc_pkl_path).parent / "100samples" / graph_doc_filename
    )

if args.style:
    graphdoc_pkl_path = (
        Path(graphdoc_pkl_path).parent / f"style{args.style}" / graph_doc_filename
    )
if args.simple:
    graphdoc_pkl_path = Path(graphdoc_pkl_path).parent / "simple" / graph_doc_filename
else:
    graphdoc_pkl_path = Path(graphdoc_pkl_path).parent / "complex" / graph_doc_filename

if args.target != "both":
    graphdoc_pkl_paths = [graphdoc_pkl_path]
else:
    ppi_graphdoc_pkl_path = Path(graphdoc_pkl_path).parent / "ppi_graph_documents.pkl"
    tf_graphdoc_pkl_path = Path(graphdoc_pkl_path).parent / "tf_graph_documents.pkl"
    graphdoc_pkl_paths = [ppi_graphdoc_pkl_path, tf_graphdoc_pkl_path]

for path in graphdoc_pkl_paths:
    graph_documents = list()

    with open(path, "rb") as f:
        while 1:
            try:
                graph_documents.append(pickle.load(f))
            except EOFError:
                break

    rels = [
        (r, g.source.page_content) for g in graph_documents for r in g.relationships
    ]
    rel_triples = [[[r.source.id, r.type, r.target.id], p] for (r, p) in rels]

    triple_filename = "triples.json"

    if args.target == "both":
        if "ppi" in path.stem:
            triple_filename = "ppi_triples.json"
        else:
            triple_filename = "tf_triples.json"

    triple_path = (
        Path(
            f"/beegfs/prj/LINDA_LLM/outputs/graph_triples/{args.target}/{args.parser}/{model}"
        )
        / triple_filename
    )

    if args.curated:
        triple_path = Path(triple_path).parent / "5curated" / triple_filename
    else:
        triple_path = Path(triple_path).parent / "100samples" / triple_filename

    if args.doclevel:
        triple_path = Path(triple_path).parent / "docs" / triple_filename
    else:
        triple_path = Path(triple_path).parent / "chunks" / triple_filename
    if args.style:
        triple_path = Path(triple_path).parent / f"style{args.style}" / triple_filename
    if args.simple:
        triple_path = Path(triple_path).parent / "simple" / triple_filename
    else:
        triple_path = Path(triple_path).parent / "complex" / triple_filename

    os.makedirs(Path(triple_path).parent, exist_ok=True)

    with open(triple_path, "w") as f:
        json.dump(rel_triples, f, indent=4)

    print(f"{triple_path}")
