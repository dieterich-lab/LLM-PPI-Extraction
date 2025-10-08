import json
import os
import sys
from pathlib import Path

sys.path.append("..")  # isort:skip
from parser import args

os.environ["BAML_LOG"] = args.loglevel  # isort:skip
from baml.baml_client.sync_client import b  # isort:skip
from baml.baml_client.types import Entities, Message, Triples  # isort:skip
from baml_py import Collector

from baml.baml_client.type_builder import TypeBuilder
from clients import cr
from documents import all_ner_paths, texts, true_ner_paths
from paths import triple_jsonl_path, uniprot_path
from prompts import prompts, rel_system_prompt

collector = Collector(name="my-collector")
tb = TypeBuilder()
if not args.noconfidence:
    tb.Triple.add_property(
        "confidence", tb.union([tb.literal_string("high"), tb.literal_string("low")])
    ).description("if this relation was extracted with high confidence or not")

if args.chattype == "lookup":
    with open(uniprot_path, "r") as lookupfile:
        lookup_table = {
            line.split("\t")[0]: (line.split("\t")[1], line.split("\t")[2])
            for line in lookupfile.readlines()[1:]
        }

if args.dynex:
    from datasets import concatenate_datasets

    from dataset import get_dataset
    from embed import client, embed_model, load_index

    index = load_index()
    train_dataset, dev_dataset, _ = get_dataset()
    lookup_dataset = concatenate_datasets([train_dataset, dev_dataset])

print(f"New run: {triple_jsonl_path.parent}")
print(f"Len texts: {len(texts)}")

if args.all_ners_given:
    ner_paths = all_ner_paths
if args.true_ners_given:
    ner_paths = true_ner_paths


def get_ners(messages, responses, doc, prompts):
    ner_prompt = prompts.pop(0)
    message = Message(role="user", content=ner_prompt)
    messages.append(message)
    try:
        ner_path = [
            x for x in ner_paths if doc[0].metadata["file_path"].stem == x.stem
        ][0]
        if ner_path:
            ners = open(ner_path, "r").readlines()
            ners = [x.strip() for x in ners]
        else:
            ners = []
        response = Entities(entities=ners)
    except:
        print(f"Exception at Entity extraction")
        response = Entities(entities=[])
    responses.append(response)
    messages.append(Message(role="assistant", content=f"{str(response)}"))
    return prompts


def extract_ners(messages, responses, text, prompts):
    ner_prompt = prompts.pop(0)
    message = Message(role="user", content=ner_prompt)
    messages.append(message)
    try:
        response = b.ExtractNEs(
            rel_system_prompt,
            text,
            message,
            baml_options={"client_registry": cr, "tb": tb, "collector": collector},
        )
    except:
        print(f"Exception at Entity extraction")
        response = Entities(entities=[])
    responses.append(response)
    messages.append(Message(role="assistant", content=f"{str(response)}"))
    return prompts


def extract_rels(messages, responses, text, prompts):
    """Standard single-pass extraction"""
    for i, prompt in enumerate(prompts):
        messages.append(Message(role="user", content=f"\nUSER QUESTION: {prompt}"))
        try:
            response = b.GeneralChatExtractRelationships(
                rel_system_prompt,
                text,
                messages,
                baml_options={"client_registry": cr, "tb": tb, "collector": collector},
            )
        except Exception as e:
            print(f"Exception at step {i}")
            response = Triples(triples=[])
        responses.append(response)
        messages.append(Message(role="assistant", content=str(response)))


def extract_rels_ensemble(
    messages, responses, text, prompts, n_samples=5, temperature=0.7
):
    """Self-consistency ensemble extraction with voting"""
    print(f"  Running ensemble extraction with n={n_samples}, temp={temperature}")

    for i, prompt in enumerate(prompts):
        all_triples = []

        # Generate n predictions with temperature > 0
        for sample_idx in range(n_samples):
            # Create a copy of messages for each sample
            messages_copy = messages.copy()
            messages_copy.append(
                Message(role="user", content=f"\nUSER QUESTION: {prompt}")
            )

            try:
                response = b.GeneralChatExtractRelationships(
                    rel_system_prompt,
                    text,
                    messages_copy,
                    baml_options={
                        "client_registry": cr,
                        "tb": tb,
                        "collector": collector,
                        "temperature": temperature,
                    },
                )
                all_triples.extend(response.triples)
                print(
                    f"    Sample {sample_idx + 1}/{n_samples}: {len(response.triples)} triples"
                )
            except Exception as e:
                print(f"    Exception at sample {sample_idx + 1}: {e}")
                continue

        # Vote: keep triples appearing in ≥50% of samples
        triple_counts = {}
        for triple in all_triples:
            # Create a unique key for each triple (case-insensitive to handle variations)
            key = (
                f"{triple.head.lower()}|{triple.relation.lower()}|{triple.tail.lower()}"
            )
            if key not in triple_counts:
                triple_counts[key] = {"count": 0, "example": triple}
            triple_counts[key]["count"] += 1

        # Select triples that appear in at least half of the samples
        threshold = n_samples // 2
        consensus_triples = [
            data["example"]
            for key, data in triple_counts.items()
            if data["count"] >= threshold
        ]

        print(
            f"    Consensus: {len(consensus_triples)} triples (threshold: {threshold}/{n_samples})"
        )

        # Create response with consensus triples
        consensus_response = Triples(triples=consensus_triples)
        responses.append(consensus_response)
        messages.append(Message(role="user", content=f"\nUSER QUESTION: {prompt}"))
        messages.append(Message(role="assistant", content=str(consensus_response)))


def lookup_infos(messages, responses):
    infos = dict()
    ent_set = set()
    for nes in responses[-1]:
        for ne in nes[1]:
            if ne in ent_set:
                continue
            ent_set.add(ne)
            if ne.lower() in lookup_table:
                infos[ne] = f"Function: {lookup_table[ne.lower()][0].strip()}"
    messages.append(Message(role="user", content=f"BACKGROUND KNOWLEDGE: {infos}\n"))


def get_dynex(messages, text):
    embed = client.embed(
        model=embed_model,
        input=text,
    ).embeddings
    labels, _ = index.knn_query(embed, k=1)
    example = lookup_dataset[labels[0]]
    example_text = "\n".join([example["doc"][0], example["triples"][0]])
    messages.append(Message(role="user", content=f"EXAMPLE: {example_text}\n"))


def main():
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
        if args.all_ners_given or args.true_ners_given:
            _prompts = get_ners(messages, responses, doc, _prompts)
        elif args.extractionmode == "nerrel":
            _prompts = extract_ners(messages, responses, text, doc, _prompts)
        if args.chattype == "lookup":
            lookup_infos(messages, responses)
        if args.dynex:
            get_dynex(messages, text)

        # Choose extraction method based on ensemble flag
        if args.ensemble:
            extract_rels_ensemble(
                messages,
                responses,
                text,
                _prompts,
                n_samples=args.ensemble,
                temperature=args.ensemble_temp,
            )
        else:
            extract_rels(messages, responses, text, _prompts)

        if not args.dev:
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
