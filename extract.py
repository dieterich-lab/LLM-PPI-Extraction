import os
import pickle
import sys

sys.path.append("..")  # isort:skip
from parser import args

os.environ["BAML_LOG"] = args.loglevel  # isort:skip
from baml.baml_client.sync_client import b  # isort:skip
from baml.baml_client.types import Entities, Message, Triples  # isort:skip
from clients import cr
from converter import convert_and_save_to_json
from documents import all_ner_paths, chunks, docs
from paths import triple_json_path, triple_pkl_path
from prompts import prompts, system_prompt

texts = docs if args.doclevel == "docs" else chunks

print(f"New run: {triple_pkl_path.parent}")


def extract_ners(messages, responses):
    ner_prompt = prompts.pop(0)
    message = Message(role="user", content=ner_prompt)
    messages.append(message)
    try:
        if not args.all_ners_given:
            response = b.ExtractNEs(
                system_prompt,
                text,
                message,
                {"client_registry": cr},
            )
        else:
            ner_path = [
                x for x in all_ner_paths if doc[0].metadata["file_path"].stem == x.stem
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


def extract_rels(messages, responses):
    for i, prompt in enumerate(prompts):
        messages.append(Message(role="user", content=prompt))
        try:
            response = b.GeneralChatExtractRelationships(
                system_prompt, text, messages, {"client_registry": cr}
            )
        except:
            print(f"Exception at step {i}")
            response = Triples(triples=[])
        responses.append(response)
        messages.append(Message(role="assistant", content=str(response)))


with open(triple_pkl_path, "wb") as triple_pkl_file:
    for i, doc in enumerate(texts):
        print(f"Doc {i}")
        text = doc[0].page_content
        messages: list[Message] = []
        responses = list()
        if args.extractionmode == "nerrel":
            extract_ners(messages, responses)
        extract_rels(messages, responses)
        pickle.dump(
            (responses, doc[0].page_content, doc[0].metadata["file_path"]),
            triple_pkl_file,
        )

        if args.dev:
            break

convert_and_save_to_json(triple_pkl_path, triple_json_path)
