import json
from parser import args

from clients import model
from ollama import Client
from paths import triple_json_path

ip_dict = {
    "g4": "10.250.135.153",
    "g2": "10.250.135.143",
    "g3": "10.250.135.150",
    "g5": "10.250.135.156",
    "mk22d": "10.250.135.115",
}


client = Client(
    host=f"http://{ip_dict[args.node]}:114{args.port}",
)

with open(triple_json_path, "r") as f:
    all_triples = json.load(f)

prompt = "Give me 10 short names for the following protein. Just answer with a list in Python format, nothing more. For example: [protein_name1, protein_name2, ...]"


d = dict()

for i, data in enumerate(all_triples):
    print(i)
    triples = data["triples"][0]
    for triple in triples:
        if triple["head"] not in d:
            head_names = client.generate(
                model=model, prompt=f"{prompt} {triple['head']}"
            )
            try:
                d[triple["head"]] = eval(head_names.response)
            except:
                pass
        if triple["tail"] not in d:
            tail_names = client.generate(
                model=model, prompt=f"{prompt} {triple['tail']}"
            )
            try:
                d[triple["tail"]] = eval(tail_names.response)
            except:
                pass

with open(triple_json_path.parent / "names.json", "w") as f:
    json.dump(d, f, indent=4)
    print(f"Saved json to {triple_json_path.parent / 'names.json'}")
