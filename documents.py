import json
import os
import pickle
from parser import args
from pathlib import Path

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
all_ner_paths = None
true_ner_paths = None
if args.data == "biored":
    _paper_paths = Path(
        "/beegfs/prj/LINDA_LLM/CardioPriorKnowledge/test_ppi_annotations/biored_26_11_2024/src/corpus"
    )
elif args.data == "regulatome":
    _paper_paths = Path(
        "/beegfs/prj/LINDA_LLM/CardioPriorKnowledge/test_ppi_annotations/regulatome_extraction_13_12_2024/src/corpus"
    )
    if args.target == "ppi":
        _all_ner_paths = Path(
            "/beegfs/prj/LINDA_LLM/CardioPriorKnowledge/test_ppi_annotations/regulatome_extraction_13_12_2024/src/entities"
        )
        all_ner_paths = list(_all_ner_paths.glob(f"*"))
        _true_ner_paths = Path(
            "/beegfs/prj/LINDA_LLM/CardioPriorKnowledge/test_ppi_annotations/regulatome_extraction_13_12_2024/src/entities_relations_ppi"
        )
        true_ner_paths = list(_true_ner_paths.glob(f"*"))
    elif args.target == "tf":
        _all_ner_paths = Path(
            "/beegfs/prj/LINDA_LLM/CardioPriorKnowledge/test_ppi_annotations/regulatome_extraction_13_12_2024/src/entities"
        )
        all_ner_paths = list(_all_ner_paths.glob(f"*"))
        _true_ner_paths = Path(
            "/beegfs/prj/LINDA_LLM/CardioPriorKnowledge/test_ppi_annotations/regulatome_extraction_13_12_2024/src/entities_relations_tf"
        )
        true_ner_paths = list(_true_ner_paths.glob(f"*"))
elif args.data == "ours":
    _paper_paths = Path(
        f"/beegfs/prj/LINDA_LLM/outputs/parsed_papers/{args.target}/{args.parser}"
    )
elif args.data == "5curated":
    _paper_paths = Path(
        f"/beegfs/prj/LINDA_LLM/outputs/parsed_papers/ppi/{args.parser}/5curated/"
    )
paper_paths = list(_paper_paths.glob(f"*.{ending_dict[args.parser]}"))

chunk_pkl_path = Path(
    f"/beegfs/prj/LINDA_LLM/outputs/docs/{args.data}/{args.target}/{args.parser}/paper_chunks.pkl"
)
os.makedirs(chunk_pkl_path.parent, exist_ok=True)
paper_pkl_path = chunk_pkl_path.parent / "papers.pkl"
os.makedirs(paper_pkl_path.parent, exist_ok=True)

paper_dict_path = Path(
    f"/beegfs/prj/LINDA_LLM/outputs/paper_dicts/{args.data}/{args.target}/{args.parser}/paper_dict.pkl"
)
os.makedirs(Path(paper_dict_path).parent, exist_ok=True)

f = open(chunk_pkl_path, "wb")
wf = open(paper_pkl_path, "wb")
with open(chunk_pkl_path, "wb") as chunk_file, open(paper_pkl_path, "wb") as doc_file:
    for i, x in enumerate(paper_paths):
        paper_dict[i] = str(x)
        text = open(x, "r").read().strip()
        if text:
            texts = text_splitter.create_documents([text])
            doc = Document(page_content=text, metadata={"file_path": x})
            for chunk in texts:
                chunk.metadata = {"file_path": x}
                pickle.dump((chunk, i), chunk_file)
            pickle.dump((doc, i), doc_file)

with open(paper_dict_path, "w") as f:
    json.dump(paper_dict, f, indent=4)

chunks = list()
with open(chunk_pkl_path, "rb") as f:
    while 1:
        try:
            chunks.append(pickle.load(f))
        except EOFError:
            break

docs = list()
with open(paper_pkl_path, "rb") as f:
    while 1:
        try:
            docs.append(pickle.load(f))
        except EOFError:
            break

texts = docs if args.doclevel == "docs" else chunks
