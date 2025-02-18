import os
from parser import args
from pathlib import Path

experiment_path = Path(
    f"/beegfs/prj/LINDA_LLM/outputs/triples/{args.data}/{args.target}/{args.model}/{args.extractionmode}/{args.chattype}/{args.doclevel}"
)
if args.all_ners_given:
    experiment_path /= "all_ners_given"
os.makedirs(experiment_path, exist_ok=True)

triple_pkl_path = experiment_path / "triples.pkl"
triple_json_path = experiment_path / "triples.json"
