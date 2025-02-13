import pickle
import sys

sys.path.append("..")
from parser import args

from baml.baml_client.sync_client import b
from baml.baml_client.types import Message
from clients import cr
from converter import convert_and_save_to_json
from documents import chunks, docs
from paths import triple_json_path, triple_pkl_path
from prompts import prompts, system_prompt

texts = docs if args.doclevel == "docs" else chunks

print(f"New run: {triple_pkl_path.parent}")

with open(triple_pkl_path, "wb") as triple_pkl_file:
    for doc in texts:
        text = doc[0].page_content
        messages: list[Message] = []
        responses = list()
        # while 1:
        try:
            if args.extractionmode == "nerrel":
                message = Message(role="user", content=prompts.pop(0))
                messages.append(message)
                response = b.ExtractNEs(
                    system_prompt,
                    text,
                    message,
                    {"client_registry": cr},
                )
                responses.append(response)
                messages.append(Message(role="assistant", content=str(response)))
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
                print(i, response)
        except:
            pass

        pickle.dump(
            (responses, doc[0].page_content, doc[0].metadata["file_path"]),
            triple_pkl_file,
        )

        if args.dev:
            break

convert_and_save_to_json(triple_pkl_path, triple_json_path)
