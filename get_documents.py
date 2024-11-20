import json
import os
import pickle
from parser import args
from pathlib import Path

from const import PAPERS
from langchain_core.documents.base import Document
from langchain_text_splitters import MarkdownTextSplitter

ending_dict = {"marker": "md", "llama_parse": "txt"}

text_splitter = MarkdownTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    length_function=len,
    is_separator_regex=False,
)
paper_dict = dict()

_paper_paths = Path(
    f"/beegfs/prj/LINDA_LLM/outputs/parsed_papers/{PAPERS}/{args.parser}"
)
if args.curated:
    _paper_paths = _paper_paths / "5curated"

paper_paths = list(_paper_paths.glob(f"*.{ending_dict[args.parser]}"))

paper_pkl_path = Path(
    f"/beegfs/prj/LINDA_LLM/outputs/paper_chunks/{PAPERS}/{args.parser}/paper_chunks.pkl"
)
if args.curated:
    paper_pkl_path = paper_pkl_path.parent / "5curated" / "paper_chunks.pkl"

paper_dict_path = Path(
    f"/beegfs/prj/LINDA_LLM/outputs/paper_dicts/{PAPERS}/{args.parser}/paper_dict.pkl"
)
if args.curated:
    paper_dict_path = paper_dict_path.parent / "5curated" / "paper_chunks.pkl"

os.makedirs(paper_pkl_path.parent, exist_ok=True)

whole_paper_pkl_path = paper_pkl_path.parent / "whole_papers.pkl"
os.makedirs(whole_paper_pkl_path.parent, exist_ok=True)

f = open(paper_pkl_path, "wb")
wf = open(whole_paper_pkl_path, "wb")
for i, x in enumerate(paper_paths):
    # print(i, x)
    paper_dict[i] = str(x)
    text = open(x, "r").read().strip()
    if text:
        texts = text_splitter.create_documents([text])
        whole_text = Document(page_content=text)
        for t in texts:
            pickle.dump((t, i), f)
        pickle.dump((whole_text, i), wf)
f.close()
wf.close()

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

whole_documents = list()
with open(whole_paper_pkl_path, "rb") as f:
    while 1:
        try:
            whole_documents.append(pickle.load(f))
        except EOFError:
            break
