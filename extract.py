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

with open(triple_pkl_path, "wb") as triple_pkl_file:
    for doc in texts:
        text = doc[0].page_content
        messages: list[Message] = []
        while 1:
            # try:
            if args.extractionmode == "nerrel":
                message = Message(role="user", content=prompts.pop(0))
                messages.append(message)
                response = b.ExtractNEs(
                    system_prompt,
                    text,
                    message,
                    {"client_registry": cr},
                )
                messages.append(Message(role="assistant", content=str(response)))
            for prompt in prompts:
                messages.append(Message(role="user", content=prompt))
                response = b.GeneralChatExtractRelationships(
                    system_prompt,
                    text,
                    messages,
                    {"client_registry": cr},
                )
                messages.append(Message(role="assistant", content=str(response)))
        # except:
        #     response = []

        pickle.dump(
            (response, doc[0].page_content, doc[0].metadata["file_path"]),
            triple_pkl_file,
        )

convert_and_save_to_json(triple_pkl_path, triple_json_path)
