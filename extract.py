import os
import pickle
import sys

sys.path.append("..")
from parser import args

os.environ["BAML_LOG"] = args.loglevel

from baml.baml_client.sync_client import b
from baml.baml_client.types import Message
from clients import cr
from converter import convert_and_save_to_json
from documents import all_ner_paths, chunks, docs
from paths import triple_json_path, triple_pkl_path
from prompts import prompts, system_prompt

texts = docs if args.doclevel == "docs" else chunks

# print(f"New run: {triple_pkl_path.parent}")

if args.extractionmode == "nerrel":
    ner_prompt = prompts.pop(0)

with open(triple_pkl_path, "wb") as triple_pkl_file:
    for doc in texts:
        try:
            text = doc[0].page_content
            messages: list[Message] = []
            responses = list()
            # while 1:
            # try:
            if args.extractionmode == "nerrel":
                message = Message(role="user", content=ner_prompt)
                messages.append(message)
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
                        Message(role="assistant", content=f" ENTITIY LIST: {str(ners)}")
                    )
            for i, prompt in enumerate(prompts):
                messages.append(Message(role="user", content=prompt))
                response = b.GeneralChatExtractRelationships(
                    system_prompt,
                    text,
                    messages,
                    {"client_registry": cr},
                )
                responses.append(response)
                messages.append(Message(role="assistant", content=str(response)))
                # print(f"STEP {i}, {response}")

            pickle.dump(
                (responses, doc[0].page_content, doc[0].metadata["file_path"]),
                triple_pkl_file,
            )
        except:
            pass

        if args.dev:
            break

convert_and_save_to_json(triple_pkl_path, triple_json_path)
