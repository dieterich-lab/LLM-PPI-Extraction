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
from paths import (
    alignment_json_path,
    corrector_json_path,
    corrector_pkl_path,
    judge_json_path,
    judge_pkl_path,
)
from prompts import (
    corrector_prompt,
    corrector_system_prompt,
    judge_prompt,
    judge_system_prompt,
)

system_prompt = (
    judge_system_prompt if args.re_evaluate == "judge" else corrector_system_prompt
)
prompt = judge_prompt if args.re_evaluate == "judge" else corrector_prompt
function = b.Judge if args.re_evaluate == "judge" else b.Correct
pkl_path = judge_pkl_path if args.re_evaluate == "judge" else corrector_pkl_path
json_path = judge_json_path if args.re_evaluate == "judge" else corrector_json_path

print(f"New run: {json_path}")


def main():
    with open(alignment_json_path, "rb") as f:
        alignments = json.load(f)
    print(f"Len alignments: {len(alignments)}")
    mode = "wb" if not args.dev else "rb"
    with open(pkl_path, mode) as pkl_file:
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
                response = function(
                    system_prompt,
                    doc,
                    reasoning,
                    str(triple),
                    prompt,
                    {"client_registry": cr},
                )
                if args.re_evaluate == "judge":
                    answer = response.reason
                elif args.re_evaluate == "corrector":
                    answer = response.judgement
                alignment.append(answer)
            except Exception as e:
                print(f"Exception at step {i}")
                response = "##Exception##"
                alignment.append(response)
            if not args.dev:
                pickle.dump(alignment, pkl_file)
            if args.dev:
                break

    with open(pkl_path, "rb") as pkl_file:
        alignments = list()
        while 1:
            try:
                alignment = pickle.load(pkl_file)
                alignments.append(alignment)
            except EOFError:
                break

    mode = "w" if not args.dev else "r"
    with open(json_path, mode) as json_file:
        json.dump(alignments, json_file, indent=4)
        print(f"Saved json to {json_path}")


if __name__ == "__main__":
    main()
