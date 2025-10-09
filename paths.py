import os
from parser import args
from pathlib import Path

from clients import hf_model_id

experiment_path = Path(
    f"/beegfs/prj/LINDA_LLM/outputs/triples/{args.data}/{args.target}/{args.model}/{args.extractionmode}/{args.chattype}/{args.doclevel}"
)
if args.doclevel == "chunks":
    experiment_path /= f"{args.chunksize}"
if args.examples:
    experiment_path /= f"{args.examples}_ex"
if args.recall:
    experiment_path /= "recall"
if args.tot:
    experiment_path /= f"tot_n{args.tot}_{args.tot_strategy}"
if args.ensemble:
    experiment_path /= f"ensemble_n{args.ensemble}_t{args.ensemble_temp}"
if args.dynex:
    experiment_path /= "dynex"

if args.all_ners_given:
    experiment_path /= "all_ners_given"
elif args.true_ners_given:
    experiment_path /= "true_ners_given"
os.makedirs(experiment_path, exist_ok=True)

if not args.ext:
    triple_jsonl_path = experiment_path / "triples.jsonl"
    triple_json_path = experiment_path / "triples.json"
else:
    triple_jsonl_path = experiment_path / f"triples_{args.ext}.jsonl"
    triple_json_path = experiment_path / f"triples_{args.ext}.json"


# try:
#     fp_paths = {
#         "deepseek8b": {
#             "nerrel": "/beegfs/prj/LINDA_LLM/RegulaTome/test_ppi_annotations/FalsePositives_DeepSeek8b_Stepwise_NoEntities_Step1_AllRel.txt",
#             "all_ners_given": "/beegfs/prj/LINDA_LLM/RegulaTome/test_ppi_annotations/FalsePositives_DeepSeek8b_Stepwise_AllEntities_Step1_AllRel.txt",
#         }
#     }
# except:
#     pass

# lookup = "nerrel" if not args.all_ners_given else "all_ners_given"
# try:
#     fp_path = fp_paths[args.model][lookup]
# except:
#     pass

# mode = "direct" if not args.all_ners_given else "all_ners_given"
# alignment_json_path = Path(
#     f"/prj/LINDA_LLM/outputs/evaluations/FPs/{args.data}_{args.model}_{mode}_{args.chattype}_{args.doclevel}.json"
# )
# os.makedirs(alignment_json_path.parent, exist_ok=True)

# judge_path = Path("/prj/LINDA_LLM/outputs/evaluations/FPs_judged")
# os.makedirs(judge_path, exist_ok=True)
# judge_pkl_path = (
#     judge_path / f"{args.data}_{args.model}_{mode}_{args.chattype}_{args.doclevel}.pkl"
# )
# judge_json_path = (
#     judge_path / f"{args.data}_{args.model}_{mode}_{args.chattype}_{args.doclevel}.json"
# )

# corrector_path = Path("/prj/LINDA_LLM/outputs/evaluations/FPs_corrected")
# os.makedirs(corrector_path, exist_ok=True)
# corrector_pkl_path = (
#     corrector_path
#     / f"{args.data}_{args.model}_{mode}_{args.chattype}_{args.doclevel}.pkl"
# )
# corrector_json_path = (
#     corrector_path
#     / f"{args.data}_{args.model}_{mode}_{args.chattype}_{args.doclevel}.json"
# )

finetune_data_path = Path("/prj/LINDA_LLM/outputs/datasets")
regulatome_ppi_eval_path = "/beegfs/prj/LINDA_LLM/RegulaTome/test_ppi_annotations/annotated_ppi_relations_dedup.txt"
regulatome_tf_eval_path = "/beegfs/prj/LINDA_LLM/RegulaTome/test_tf_annotations/annotated_tf_relations_dedup_new.txt"
try:
    sft_model_path = (
        Path("/prj/LINDA_LLM/outputs") / "finetunedmodels" / f"{hf_model_id}_regulatome"
    )
except TypeError:
    pass

uniprot_path = "/prj/LINDA_LLM/resources/uniprot_description_and_interactors.txt"
