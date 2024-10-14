import json
import os
import pickle
from pathlib import Path

# from langchain_community.graphs import Neo4jGraph

# task = "ppi"
task = "tf"
old = True

# parser = "marker"
parser = "llama_parse"

model = "llama3.1:70b"
# model = "llama3.1:8b"

task_dict = {"ppi": "linda-llm", "tf": "linda-llm-dev"}

graph_documents = list()
if not old:
    graphdoc_pkl_path = f"/beegfs/prj/LINDA_LLM/outputs/graph_docs/{task}/{parser}/{model}/graph_documents.pkl"
else:
    graphdoc_pkl_path = f"/beegfs/prj/LINDA_LLM/outputs/graph_docs/tf/{parser}/{model}/graph_documents_old.pkl"


print(f"loading from {graphdoc_pkl_path}")
with open(graphdoc_pkl_path, "rb") as f:
    while 1:
        try:
            graph_documents.append(pickle.load(f))
        except EOFError:
            break

print(len(graph_documents))

rels = [r for g in graph_documents for r in g.relationships]
rel_triples = [(r.source.id, r.type, r.target.id) for r in rels]

if not old:
    triple_path = f"/home/pwiesenbach/LINDA_LLM/outputs/graph_triples/{task}/{parser}/{model}/triples.json"
else:
    triple_path = f"/home/pwiesenbach/LINDA_LLM/outputs/graph_triples/tf/{parser}/{model}/triples_old.json"

os.makedirs(Path(triple_path).parent, exist_ok=True)

with open(triple_path, "w") as f:
    json.dump(rel_triples, f, indent=4)
