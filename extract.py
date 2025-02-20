import os
import pickle
import sys

sys.path.append("..")  # isort:skip
from parser import args  # isort:skip

os.environ["BAML_LOG"] = args.loglevel
from baml.baml_client.sync_client import b  # isort:skip
from baml.baml_client.types import Entities, Message, Triples  # isort:skip
from clients import cr
from converter import convert_and_save_to_json
from documents import all_ner_paths, chunks, docs
from paths import triple_json_path, triple_pkl_path
from prompts import prompts, system_prompt

texts = docs if args.doclevel == "docs" else chunks

print(f"New run: {triple_pkl_path.parent}")

if args.extractionmode == "nerrel":
    ner_prompt = prompts.pop(0)

with open(triple_pkl_path, "wb") as triple_pkl_file:
    for i, doc in enumerate(texts):
        print(f"Doc {i}")
        text = doc[0].page_content
        messages: list[Message] = []
        responses = list()
        if args.extractionmode == "nerrel":
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
                    responses.append(response)
                    messages.append(
                        Message(
                            role="assistant", content=f" ENTITY LIST: {str(response)}"
                        )
                    )
                else:
                    ner_path = [
                        x
                        for x in all_ner_paths
                        if doc[0].metadata["file_path"].stem == x.stem
                    ][0]
                    if ner_path:
                        ners = open(ner_path, "r").readlines()
                        ners = [x.strip() for x in ners]
                    else:
                        ners = []
                    responses.append(ners)
                    messages.append(
                        Message(
                            role="assistant",
                            content=f" ENTITIY LIST: {Entities(entities=ners)}",
                        )
                    )
            except:
                print(f"Exception at Entity extraction")
                responses.append(Entities(entities=[]))
                messages.append(
                    Message(
                        role="assistant",
                        content=f" ENTITIY LIST: {Entities(entities=[])}",
                    )
                )
        for j, prompt in enumerate(prompts):
            messages.append(Message(role="user", content=prompt))
            try:
                response = b.GeneralChatExtractRelationships(
                    system_prompt,
                    text,
                    messages,
                    {"client_registry": cr},
                )
                responses.append(response)
                messages.append(Message(role="assistant", content=str(response)))
            except:
                print(f"Exception at step {j}")
                responses.append(Triples(triples=[]))
                messages.append(
                    Message(role="assistant", content=str(Triples(triples=[])))
                )

        pickle.dump(
            (responses, doc[0].page_content, doc[0].metadata["file_path"]),
            triple_pkl_file,
        )

        if args.dev:
            break

convert_and_save_to_json(triple_pkl_path, triple_json_path)
