import json
import os
from parser import args
from pathlib import Path

from documents import all_docs
from paths import (
    finetune_data_path,
    regulatome_ppi_eval_path,
    regulatome_tf_eval_path,
    triple_json_path,
)

os.environ["BAML_LOG"] = args.loglevel  # isort:skip
from baml.baml_client.sync_client import b  # isort:skip
from clients import cr

with open(triple_json_path, "r") as f:
    all_triples = json.load(f)

from dataset import get_dataset

datasets = [*get_dataset(target=args.target)]

d = dict()

data_path = (
    regulatome_ppi_eval_path if args.target == "ppi" else regulatome_tf_eval_path
)
with open(data_path, "r") as f:
    data = [
        (x.split("\t")[0], x.split("\t")[1], x.split("\t")[2].strip())
        for x in f.readlines()[1:]
    ]

data = [{"file_stem": x[0], "relations": x[1], "split": x[2]} for x in data]

relations = [x["relations"].split(";") for x in data]
relations = [
    [(x.split("=")[0].strip(), x.split("=")[1].strip()) for x in rel]
    for rel in relations
]

for rel in relations:
    for (
        head,
        tail,
    ) in rel:
        if head not in d:
            try:
                alt_names = b.CreateAltNames(
                    head,
                    {"client_registry": cr},
                ).alt_names
                d[head] = alt_names
            except:
                pass
        if tail not in d:
            try:
                alt_names = b.CreateAltNames(
                    tail,
                    {"client_registry": cr},
                ).alt_names
                d[tail] = alt_names
            except:
                pass

path = Path(f"/prj/LINDA_LLM/outputs/synonyms/{args.target}/{args.model}")
path.mkdir(parents=True, exist_ok=True)
with open(path / "synonyms.json", "w") as f:
    json.dump(d, f, indent=4)

print(f"Wrote synonyms for {path}.")

# with open(triple_json_path, "r") as f:
#     all_triples = json.load(f)

# d = dict()

# for i, data in enumerate(all_triples):
#     print(i)
#     triples = data["triples"][0]
#     for triple in triples:
#         if triple["head"] not in d:
#             try:
#                 alt_names = b.CreateAltNames(
#                     triple["head"],
#                     {"client_registry": cr},
#                 ).alt_names
#                 d[triple["head"]] = alt_names
#             except:
#                 pass
#         if triple["tail"] not in d:
#             try:
#                 alt_names = b.CreateAltNames(
#                     triple["tail"],
#                     {"client_registry": cr},
#                 ).alt_names
#                 d[triple["tail"]] = alt_names
#             except:
#                 pass

# with open(triple_json_path.parent / "names.json", "w") as f:
#     json.dump(d, f, indent=4)
#     print(f"Saved json to {triple_json_path.parent / 'names.json'}")
