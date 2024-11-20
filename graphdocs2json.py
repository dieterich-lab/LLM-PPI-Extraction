import json
import pickle
from parser import args

from paths import graphdoc_pkl_paths, triple_path

for i, path in enumerate(graphdoc_pkl_paths):
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
        if i == 0:
            triple_filename = "ppi_triples.json"
        else:
            triple_filename = "tf_triples.json"

    with open(triple_path / triple_filename, "w") as f:
        json.dump(rel_triples, f, indent=4)

    print(f"{triple_path / triple_filename}")
