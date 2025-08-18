import os
import pickle
import sys
import tempfile

sys.path.append("..")  # isort:skip
from parser import args

os.environ["BAML_LOG"] = args.loglevel  # isort:skip
from baml.baml_client.sync_client import b  # isort:skip
from baml.baml_client.types import Entities, Message, Triples  # isort:skip
from baml_py import Collector

from baml.baml_client.type_builder import TypeBuilder
from clients import cr
from converter import convert_and_save_triples_to_json
from documents import all_ner_paths, texts, true_ner_paths
from paths import triple_json_path, triple_pkl_path, uniprot_path
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

print(f"New run: {triple_pkl_path.parent}")
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


def load_file_paths(file):
    file_paths = []
    try:
        with open(file, "rb") as triple_pkl_file:
            while True:
                try:
                    tuple = pickle.load(triple_pkl_file)
                    file_path = str(tuple[2])
                    file_paths.append(file_path)
                except EOFError:
                    break
    except FileNotFoundError:
        print(f"File {file} not found, starting fresh.")
    return file_paths


def main():
    file = triple_pkl_path if not args.dev else tempfile.NamedTemporaryFile().name
    file_paths = load_file_paths(file)
    try:
        with open(file, "ab+") as triple_pkl_file:
            for i, doc in enumerate(texts):
                file_path = doc[0].metadata["file_path"]
                if not args.force_new and str(file_path) in file_paths:
                    continue
                file_paths.append(str(file_path))
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
                extract_rels(messages, responses, text, _prompts)
                if not args.dev:
                    pickle.dump(
                        (responses, doc[0].page_content, doc[0].metadata["file_path"]),
                        triple_pkl_file,
                    )
    finally:
        if not args.dev:
            convert_and_save_triples_to_json(triple_pkl_path, triple_json_path)


if __name__ == "__main__":
    main()
