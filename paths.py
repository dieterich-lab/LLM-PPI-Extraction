import os
from parser import args
from pathlib import Path

experiment_path = Path(
    f"/beegfs/prj/LINDA_LLM/outputs/triples/{args.data}/{args.target}/{args.model}/{args.extractionmode}/{args.chattype}/{args.doclevel}"
)
all_ners_given = "all_ners_given" if args.all_ners_given else ""
slurm_path = Path(
    f"/prj/LINDA_LLM/outputs/slurm/{args.data}_{args.model}{all_ners_given}.txt"
)

if args.all_ners_given:
    experiment_path /= "all_ners_given"
os.makedirs(experiment_path, exist_ok=True)

triple_pkl_path = experiment_path / "triples.pkl"
triple_json_path = experiment_path / "triples.json"

regulatome_eval_path = "/beegfs/prj/LINDA_LLM/CardioPriorKnowledge/test_ppi_annotations/annotated_ppi_relations.txt"

try:
    fp_paths = {
        "deepseek8b": {
            "nerrel": "/beegfs/prj/LINDA_LLM/CardioPriorKnowledge/test_ppi_annotations/FalsePositives_DeepSeek8b_Stepwise_NoEntities_Step1_AllRel.txt",
            "all_ners_given": "/beegfs/prj/LINDA_LLM/CardioPriorKnowledge/test_ppi_annotations/FalsePositives_DeepSeek8b_Stepwise_AllEntities_Step1_AllRel.txt",
        }
    }
except:
    pass

lookup = "nerrel" if not args.all_ners_given else "all_ners_given"
try:
    fp_path = fp_paths[args.model][lookup]
except:
    pass

mode = "direct" if not args.all_ners_given else "all_ners_given"
alignment_json_path = Path(
    f"/prj/LINDA_LLM/outputs/evaluations/FPs/{args.data}_{args.model}_{mode}_{args.chattype}_{args.doclevel}.json"
)
os.makedirs(alignment_json_path.parent, exist_ok=True)

judge_path = Path("/prj/LINDA_LLM/outputs/evaluations/FPs_judged")
os.makedirs(judge_path.parent, exist_ok=True)
judge_pkl_path = (
    judge_path / f"{args.data}_{args.model}_{mode}_{args.chattype}_{args.doclevel}.pkl"
)
judge_json_path = (
    judge_path / f"{args.data}_{args.model}_{mode}_{args.chattype}_{args.doclevel}.json"
)
