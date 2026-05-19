import json
import logging
import os
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Tuple

import click
from baml_py import ClientRegistry, Collector

from parse_pdfs import convert_pdf_to_markdown, ensure_directory_exists

DEFAULT_POS_DIR = (
    "/prj/LINDA_LLM/resources/CardiacFilterPapers/"
    "Positive_Examples-20260210T142652Z-1-001/Positive_Examples"
)
DEFAULT_NEG_DIR = (
    "/prj/LINDA_LLM/resources/CardiacFilterPapers/"
    "Negative_Examples-20260210T142630Z-1-001/Negative_Examples"
)
DEFAULT_OUTPUT_DIR = "/prj/LINDA_LLM/outputs/parsed_papers/CardiacFilter"
DEFAULT_RESULTS_PATH = "/prj/LINDA_LLM/outputs/cardiac_filter/results.jsonl"
DEFAULT_CARDIOPRIOR_PPI_DIR = "/prj/LINDA_LLM/CardioPrior/PPI_Papers"
DEFAULT_CARDIOPRIOR_OUTPUT_DIR = (
    "/prj/LINDA_LLM/outputs/parsed_papers/CardioPrior/ppi_filter"
)
DEFAULT_CARDIOPRIOR_RESULTS_PATH = (
    "/prj/LINDA_LLM/outputs/cardiac_filter/ppi_results.jsonl"
)

MODEL_CHOICES = [
    "llama31",
    "llama33",
    "llama31regu",
    "llama33regu",
    "llama31regutf",
    "llama33regutf",
    "qwen3",
    "qwen314",
    "qwen330",
    "qwen332",
]
NODE_CHOICES = ["g2", "g3", "g4", "g5", "mk22d", "local"]
LOGLEVEL_CHOICES = ["error", "warn", "info", "debug", "trace", "off"]

logger = logging.getLogger("cardiac_filter")

IP_DICT = {
    "g4": "10.250.135.153",
    "g2": "10.250.135.143",
    "g3": "10.250.135.150",
    "g5": "10.250.135.156",
    "mk22d": "10.250.135.115",
    "local": "127.0.0.1",
}

OLLAMA_CLIENTS: List[Tuple[str, str]] = [
    ("llama33", "llama3.3:70b"),
    ("llama31", "llama3.1:8b"),
    ("deepseek8b", "deepseek-r1-128k:8b"),
    ("deepseek70b", "deepseek-r1-128k:70b"),
    ("gemma", "gemma3:27b"),
    ("llama33regu", "llama3.3:70b-regu_Q4_K_M"),
    ("llama31regu", "llama3.1:8b-regu_Q4_K_M"),
    ("llama33regutf", "llama3.3-70b-regu_tf"),
    ("llama31regutf", "llama3.1-8b-regu_tf"),
    ("qwen3", "qwen3:8b"),
    ("qwen314", "qwen3:14b"),
    ("qwen330", "qwen3:30b"),
    ("qwen332", "qwen3:32b"),
]

PORT_DICT = {"g2": 32, "g3": 33, "g4": 34, "g5": 35, "local": 34}


def get_supported_files(directory: str) -> List[Path]:
    """Get all supported input files recursively from a directory."""
    root = Path(directory)
    patterns = ("**/*.pdf", "**/*.md", "**/*.txt")
    paths: List[Path] = []
    for pattern in patterns:
        paths.extend(root.glob(pattern))
    return paths


def build_client_registry(model: str, node: str, port: Optional[int]) -> ClientRegistry:
    if port is None:
        port = PORT_DICT[node]
    base_url = f"http://{IP_DICT[node]}:114{port}/v1"

    cr = ClientRegistry()
    for name, client_model in OLLAMA_CLIENTS:
        cr.add_llm_client(
            name=name,
            provider="openai-generic",
            options={
                "base_url": base_url,
                "model": client_model,
                "max_tokens": 10000,
                "temperature": 0.0,
                "n_ctx": 120_000,
            },
        )
    cr.set_primary(model)
    return cr


def normalize_answer(answer: Optional[str]) -> Optional[bool]:
    if not answer:
        return None
    normalized = answer.strip().lower()
    if normalized == "yes":
        return True
    if normalized == "no":
        return False
    return None


def infer_ground_truth(path: Path) -> Optional[str]:
    parts = {part.lower() for part in path.parts}
    if "positive_examples" in parts:
        return "cardiac"
    if "negative_examples" in parts:
        return "non-cardiac"
    return None


def collect_input_files(
    single_file: Optional[Path], input_dirs: Iterable[str]
) -> List[Path]:
    if single_file:
        return [Path(single_file)]

    input_paths: List[Path] = []
    for input_dir in input_dirs:
        input_paths.extend(get_supported_files(input_dir))
    return sorted(set(input_paths))


def load_markdown(
    path: Path, output_dir: Path, converter: Literal["docling", "pymupdf4llm"]
) -> Tuple[Optional[str], Optional[Path]]:
    if path.suffix.lower() == ".pdf":
        ensure_directory_exists(str(output_dir))
        md_path = output_dir / f"{path.stem}.md"
        final_md_path: Optional[Path] = md_path
        if not md_path.exists():
            final_md_path = convert_pdf_to_markdown(path, output_dir, converter)
        if final_md_path and final_md_path.exists():
            return final_md_path.read_text(encoding="utf-8"), final_md_path
        return None, final_md_path

    if path.suffix.lower() in {".md", ".txt"}:
        return path.read_text(encoding="utf-8"), path

    logger.warning("Skipping unsupported file type: %s", path)
    return None, None


def load_processed_paths(results_path: Path) -> set[str]:
    processed: set[str] = set()
    if not results_path.exists():
        return processed
    try:
        with open(results_path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                file_path = row.get("file_path")
                if isinstance(file_path, str):
                    processed.add(file_path)
    except OSError as exc:
        logger.warning("Could not read results file %s: %s", results_path, exc)
    return processed


def classify_markdown(
    text: str, cr: ClientRegistry, collector: Collector
) -> Dict[str, object]:
    from baml.baml_client.sync_client import b

    prompt_calls = [
        ("prompt_1", b.CardiacFilterQ1),
        ("prompt_2", b.CardiacFilterQ2),
        ("prompt_3", b.CardiacFilterQ3),
    ]

    votes = []
    errors = []
    for prompt_name, func in prompt_calls:
        try:
            response = func(
                text,
                baml_options={"client_registry": cr, "collector": collector},
            )
            answer = getattr(response, "answer", None)
            votes.append(
                {
                    "prompt": prompt_name,
                    "answer": answer,
                    "is_cardiac": normalize_answer(answer),
                }
            )
        except Exception as exc:
            errors.append(f"{prompt_name}: {exc}")
            votes.append({"prompt": prompt_name, "answer": None, "is_cardiac": None})

    yes_votes = sum(1 for vote in votes if vote["is_cardiac"] is True)
    is_cardiac = yes_votes >= 2

    return {
        "votes": votes,
        "yes_votes": yes_votes,
        "decision": "cardiac" if is_cardiac else "non-cardiac",
        "is_cardiac": is_cardiac,
        "errors": errors,
    }


@click.command()
@click.option(
    "--input-dir",
    "input_dirs",
    multiple=True,
    default=[DEFAULT_POS_DIR, DEFAULT_NEG_DIR],
    show_default=True,
    help="Directory containing PDF papers (can be set multiple times).",
)
@click.option(
    "--single-file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Process a single PDF/markdown file instead of a directory.",
)
@click.option(
    "--cardioprior-ppi",
    is_flag=True,
    help="Process the full CardioPrior PPI_Papers dataset.",
)
@click.option(
    "--output-dir",
    default=DEFAULT_OUTPUT_DIR,
    show_default=True,
    help="Directory to store parsed markdown files.",
)
@click.option(
    "--output-jsonl",
    default=DEFAULT_RESULTS_PATH,
    show_default=True,
    help="JSONL file to append results to.",
)
@click.option(
    "--skip-existing/--no-skip-existing",
    default=True,
    show_default=True,
    help="Skip papers already present in the results JSONL.",
)
@click.option(
    "--converter",
    type=click.Choice(["docling", "pymupdf4llm"]),
    default="pymupdf4llm",
    show_default=True,
    help="PDF-to-markdown converter to use.",
)
@click.option(
    "--max-chars",
    type=int,
    default=120_000,
    show_default=True,
    help="Maximum number of characters to send to the model (120k chars ≈ 30k tokens).",
)
@click.option(
    "--limit",
    type=int,
    default=0,
    show_default=True,
    help="Limit number of papers processed (0 means no limit).",
)
@click.option(
    "--num-shards",
    type=int,
    default=1,
    show_default=True,
    help="Total number of deterministic shards to split input files into.",
)
@click.option(
    "--shard-index",
    type=int,
    default=0,
    show_default=True,
    help="Zero-based shard index to process (must be < num-shards).",
)
@click.option(
    "--model",
    type=click.Choice(MODEL_CHOICES),
    default="llama31",
    show_default=True,
    help="Model alias pointing to the local Ollama server.",
)
@click.option(
    "--node",
    type=click.Choice(NODE_CHOICES),
    default="g4",
    show_default=True,
    help="Node alias where the Ollama server is running.",
)
@click.option(
    "--port",
    type=int,
    help="Port if deviating from the standard port mapping.",
)
@click.option(
    "--loglevel",
    type=click.Choice(LOGLEVEL_CHOICES),
    default="off",
    show_default=True,
    help="BAML logging level.",
)
@click.option("--verbose", is_flag=True, help="Enable debug logging.")
def main(
    input_dirs: Tuple[str, ...],
    single_file: Optional[Path],
    cardioprior_ppi: bool,
    output_dir: str,
    output_jsonl: str,
    skip_existing: bool,
    converter: Literal["docling", "pymupdf4llm"],
    max_chars: int,
    limit: int,
    num_shards: int,
    shard_index: int,
    model: str,
    node: str,
    port: Optional[int],
    loglevel: str,
    verbose: bool,
) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    os.environ["BAML_LOG"] = loglevel

    if not single_file and not input_dirs:
        raise click.UsageError("Provide --single-file or at least one --input-dir.")

    if cardioprior_ppi and not single_file:
        input_dirs = (DEFAULT_CARDIOPRIOR_PPI_DIR,)
        output_dir = DEFAULT_CARDIOPRIOR_OUTPUT_DIR
        output_jsonl = DEFAULT_CARDIOPRIOR_RESULTS_PATH

    if num_shards < 1:
        raise click.UsageError("--num-shards must be >= 1.")
    if shard_index < 0 or shard_index >= num_shards:
        raise click.UsageError("--shard-index must be in [0, num-shards).")

    cr = build_client_registry(model, node, port)
    collector = Collector(name="cardiac-filter")

    output_dir_path = Path(output_dir)
    ensure_directory_exists(str(output_dir_path))

    results_path = Path(output_jsonl) if output_jsonl else None
    if results_path:
        ensure_directory_exists(str(results_path.parent))
    processed_paths: set[str] = set()
    if results_path and skip_existing:
        processed_paths = load_processed_paths(results_path)

    paper_paths = collect_input_files(single_file, input_dirs)
    if num_shards > 1:
        paper_paths = [
            path
            for idx, path in enumerate(paper_paths)
            if idx % num_shards == shard_index
        ]
    if limit and limit > 0:
        paper_paths = paper_paths[:limit]

    logger.info("Processing %d papers", len(paper_paths))
    if num_shards > 1:
        logger.info("Shard %d/%d", shard_index + 1, num_shards)
    if processed_paths:
        logger.info("Skipping %d already-processed papers", len(processed_paths))

    for paper_path in paper_paths:
        if processed_paths and str(paper_path) in processed_paths:
            logger.info("Skipping already processed %s", paper_path)
            continue
        logger.info("Processing %s", paper_path)
        text, md_path = load_markdown(paper_path, output_dir_path, converter)
        if not text:
            logger.warning("Skipping %s (no text)", paper_path)
            continue

        if max_chars and len(text) > max_chars:
            text = text[:max_chars]

        classification = classify_markdown(text, cr, collector)
        result = {
            "file_path": str(paper_path),
            "markdown_path": str(md_path) if md_path else None,
            "ground_truth": infer_ground_truth(paper_path),
            "decision": classification["decision"],
            "is_cardiac": classification["is_cardiac"],
            "yes_votes": classification["yes_votes"],
            "votes": classification["votes"],
            "errors": classification["errors"],
        }

        if results_path:
            with open(results_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(result) + "\n")
        else:
            logger.info("Result: %s", result)


if __name__ == "__main__":
    main()
