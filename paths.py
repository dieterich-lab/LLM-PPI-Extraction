import os
from parser import args
from pathlib import Path

from clients import hf_model_id
from dotenv import load_dotenv

load_dotenv()  # load .env from cwd or any parent directory


def _env_path(env_key: str, default: Path) -> Path:
    """Resolve a path from env or fall back to default."""
    value = os.environ.get(env_key)
    if value:
        return Path(value).expanduser().resolve()
    return default


# Resolve key project directories with sensible defaults that can be overridden.
PROJECT_ROOT = _env_path("LINDA_LLM_PROJECT_ROOT", Path(__file__).resolve().parents[1])
OUTPUT_ROOT = _env_path("LINDA_LLM_OUTPUT_ROOT", PROJECT_ROOT / "outputs")
TRIPLES_ROOT = _env_path("LINDA_LLM_TRIPLES_ROOT", OUTPUT_ROOT / "triples")
REGULATOME_ROOT = _env_path("LINDA_LLM_REGULATOME_ROOT", PROJECT_ROOT / "RegulaTome")
RESOURCES_ROOT = _env_path("LINDA_LLM_RESOURCES_ROOT", PROJECT_ROOT / "resources")


experiment_path = (
    TRIPLES_ROOT
    / args.data
    / args.target
    / args.model
    / args.extractionmode
    / args.chattype
    / args.doclevel
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
if args.dynex_k > 0:
    experiment_path /= f"dynex_k{args.dynex_k}"
if args.lookup:
    experiment_path /= f"lookup"

if args.all_nes_given:
    experiment_path /= "all_nes_given"
elif args.true_nes_given:
    experiment_path /= "true_nes_given"
elif args.spacy_nes_given:
    experiment_path /= "spacy_nes_given"
os.makedirs(experiment_path, exist_ok=True)

if not args.ext:
    triple_jsonl_path = experiment_path / "triples.jsonl"
    triple_json_path = experiment_path / "triples.json"
else:
    triple_jsonl_path = experiment_path / f"triples_{args.ext}.jsonl"
    triple_json_path = experiment_path / f"triples_{args.ext}.json"

finetune_data_path = OUTPUT_ROOT / "datasets"
regulatome_ppi_eval_path = (
    REGULATOME_ROOT / "test_ppi_annotations" / "annotated_ppi_relations_dedup.txt"
)
regulatome_tf_eval_path = (
    REGULATOME_ROOT / "test_tf_annotations" / "annotated_tf_relations_dedup_new.txt"
)
try:
    sft_model_path = OUTPUT_ROOT / "finetunedmodels" / f"{hf_model_id}_regulatome"
except TypeError:
    pass

uniprot_path = RESOURCES_ROOT / "uniprot_description_and_interactors.txt"
