import json
import os
import re
import sys
from pathlib import Path

import numpy as np

sys.path.append("..")  # isort:skip
from parser import args

os.environ["BAML_LOG"] = args.loglevel  # isort:skip
from baml.baml_client.sync_client import b  # isort:skip
from baml.baml_client.types import (  # isort:skip
    Entities,
    Message,
    Triples,
)
from baml_py import Collector

from baml.baml_client.type_builder import TypeBuilder
from brat_utils import parse_brat_annotations
from clients import cr
from documents import all_nes_paths, spacy_ne_paths, texts, true_ne_paths
from extraction_utils import (
    combine_by_best_path,
    combine_by_merging,
    combine_by_voting,
    extract_nes,
    extract_rels,
    extract_rels_ensemble,
    extract_rels_tot,
    get_nes,
)
from paths import triple_jsonl_path
from prompts import (
    confidence_prompt,
    interactions_type,
    prompts,
    rel_system_prompt,
    tot_path_extraction_prompt,
)

collector = Collector(name="my-collector")
tb = TypeBuilder()
if not args.noconfidence:
    tb.Triple.add_property(
        "confidence", tb.union([tb.literal_string("high"), tb.literal_string("low")])
    ).description("if this relation was extracted with high confidence or not")

# Initialize PPI lookup components if needed
protein_to_ppis = None
synonyms = None
ix = None

if args.lookup:
    from ppi_lookup import create_whoosh_index, load_string_data, load_synonyms

    protein_to_ppis = load_string_data()
    synonyms = load_synonyms()
    ix = create_whoosh_index(protein_to_ppis)

# Initialize RAG components if needed
index = None
embeddings = None
lookup_dataset = None
client = None
embed_model = None

if args.dynex_k > 0:
    from datasets import concatenate_datasets

    from dataset import get_dataset
    from embed import client, embed_model, embeddings_path, load_index
    from rag_utils import get_dynex

    index = load_index()
    embeddings = np.load(embeddings_path)
    train_dataset, dev_dataset, _ = get_dataset(args.target, args.data)
    lookup_dataset = concatenate_datasets([train_dataset, dev_dataset])

print(f"New run: {triple_jsonl_path.parent}")
print(f"Len texts: {len(texts)}")

if args.all_nes_given:
    ne_paths = all_nes_paths
if args.true_nes_given:
    ne_paths = true_ne_paths


def main():
    examples_content = ""  # Store examples content for inclusion in prompts
    if args.force_new:
        with open(triple_jsonl_path, "w") as f:  # delete the content
            pass

    # Apply document slicing based on startfromdoc and untildoc
    docs_to_process = texts
    if args.untildoc > 0:
        docs_to_process = texts[args.startfromdoc : args.untildoc]
    elif args.startfromdoc > 0:
        docs_to_process = texts[args.startfromdoc :]

    print(
        f"Processing documents {args.startfromdoc} to {args.untildoc if args.untildoc > 0 else len(texts)} (total: {len(docs_to_process)})"
    )

    for i, doc in enumerate(docs_to_process, start=args.startfromdoc):
        file_path = doc[0].metadata["file_path"]
        if not args.force_new and not args.dev:
            try:
                with open(triple_jsonl_path, "r") as f:
                    if any(
                        Path(json.loads(line)["filename"]).stem == file_path.stem
                        for line in f
                    ):
                        print(f"Skipping {file_path.stem}")
                        continue
            except FileNotFoundError:
                pass
        _prompts = prompts.copy()
        print(f"Doc {i}")
        text = doc[0].page_content
        messages = list()
        responses = list()
        if args.all_nes_given or args.true_nes_given or args.spacy_nes_given:
            _prompts = get_nes(messages, responses, doc, _prompts, collector, tb)
        elif args.extractionmode == "nerrel":
            _prompts = extract_nes(messages, responses, text, _prompts, collector, tb)
        if args.lookup:
            from ppi_lookup import lookup_infos

            lookup_infos(messages, responses, protein_to_ppis, synonyms, ix)
        if args.dynex_k > 0:
            from rag_utils import get_dynex

            get_dynex(
                messages,
                text,
                args,
                index,
                embeddings,
                lookup_dataset,
                client,
                embed_model,
            )
            examples_content = messages[-1].content  # Get the examples content
            messages.pop()  # Remove it from messages, will be included in prompts

        # Choose extraction method based on flags
        if args.tot:
            extract_rels_tot(
                messages,
                responses,
                text,
                _prompts,
                n_paths=args.tot,
                strategy=args.tot_strategy,
                examples_content=examples_content,
                collector=collector,
                tb=tb,
            )
        elif args.ensemble:
            extract_rels_ensemble(
                messages,
                responses,
                text,
                _prompts,
                n_samples=args.ensemble,
                temperature=args.ensemble_temp,
                examples_content=examples_content,
                collector=collector,
                tb=tb,
            )
        else:
            extract_rels(
                messages,
                responses,
                text,
                _prompts,
                examples_content=examples_content,
                collector=collector,
                tb=tb,
            )

        if not args.dev:
            if (
                args.extractionmode == "nerrel"
                and not args.true_nes_given
                and not args.all_nes_given
                and not args.spacy_nes_given
            ):
                NE_list = [r.model_dump()["entities"] for r in responses[:1]]
                result = {
                    "responses": NE_list
                    + [r.model_dump()["triples"] for r in responses[1:]],
                    "text": doc[0].page_content,
                    "filename": str(doc[0].metadata["file_path"]),
                }
            else:
                result = {
                    "responses": [r.model_dump()["triples"] for r in responses],
                    "text": doc[0].page_content,
                    "filename": str(doc[0].metadata["file_path"]),
                }
            with open(triple_jsonl_path, "a") as f:
                f.write(json.dumps(result) + "\n")
                f.flush()


if __name__ == "__main__":
    main()
