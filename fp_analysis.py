import json
import re
from pathlib import Path

from paths import alignment_json_path, fp_path, slurm_path, triple_json_path

with open(triple_json_path, "r") as f:
    data = json.load(f)
filenames = [x["filename"].strip() for x in data]

with open(slurm_path, "r") as f:
    data = f.read()
docs = re.split(r"Doc \d+", data)[1:]

file_dict = {Path(filename).stem: doc for filename, doc in zip(filenames, docs)}

with open(fp_path, "r") as f:
    fps = f.readlines()[1:]

alignmnents = list()
for fp in fps:
    head, tail, filename = fp.split("\t")
    doc = file_dict[filename.replace(".txt", "").strip()]
    replies = re.findall(r"(?<=<think>)[\w\W]*?(?=</think>)", doc)
    alignment = list()
    for reply in replies:
        curated_lines = [
            line.strip()
            for line in reply.split("\n")
            if (head.lower() in line.lower() or tail.lower() in line.lower())
            and not re.match("\s+\d\.", line)
        ]
        alignment.append(curated_lines)
    alignmnents.append([head, tail, filename.strip(), alignment])

with open(alignment_json_path, "w") as f:
    json.dump(alignmnents, f, indent=4)
