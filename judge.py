import json
import os
import pickle
import sys

sys.path.append("..")  # isort:skip
from parser import args

os.environ["BAML_LOG"] = args.loglevel  # isort:skip
from baml.baml_client.sync_client import b  # isort:skip
from baml.baml_client.types import Triple  # isort:skip
from clients import cr
from documents import texts
from paths import alignment_json_path, judge_json_path, judge_pkl_path
from prompts import judge_prompt, judge_system_prompt

print(f"New run: {judge_json_path}")


def main():
    with open(alignment_json_path, "rb") as f:
        alignments = json.load(f)
    print(f"Len alignments: {len(alignments)}")
    mode = "wb" if not args.dev else "rb"
    with open(judge_pkl_path, mode) as judge_pkl_file:
        for i, alignment in enumerate(alignments):
            print(f"Alignment {i}")
            triple = Triple(
                head=alignment[0], relation="INTERACTS_WITH", tail=alignment[1]
            )
            reasoning = "\n".join(alignment[3][0])
            doc = [
                doc
                for doc in texts
                if alignment[2] in str(doc[0].metadata["file_path"])
            ][0][0].page_content
            try:
                response = b.Judge(
                    judge_system_prompt,
                    doc,
                    reasoning,
                    str(triple),
                    judge_prompt,
                    {"client_registry": cr},
                )
            except:
                print(f"Exception at step {i}")
                response = "##Exception##"
            alignment.append(response.reason)
            if not args.dev:
                pickle.dump(alignment, judge_pkl_file)
            if args.dev:
                break

    with open(judge_pkl_path, "rb") as judge_pkl_file:
        judge_alignments = list()
        while 1:
            try:
                alignment = pickle.load(judge_pkl_file)
                judge_alignments.append(alignment)
            except EOFError:
                break

    with open(judge_json_path, "w") as judge_json_file:
        json.dump(judge_alignments, judge_json_file, indent=4)
        print(f"Saved json to {judge_json_path}")


if __name__ == "__main__":
    main()
