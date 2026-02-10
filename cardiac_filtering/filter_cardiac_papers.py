import json
import logging
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import click
from baml_py import ClientRegistry, Collector

from parse_pdfs import convert_pdf_to_markdown, ensure_directory_exists, get_pdf_files

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
NODE_CHOICES = ["g2", "g3", "g4", "g5", "mk22d"]
LOGLEVEL_CHOICES = ["error", "warn", "info", "debug", "trace", "off"]

logger = logging.getLogger("cardiac_filter")

IP_DICT = {
    "g4": "10.250.135.153",
    "g2": "10.250.135.143",
    "g3": "10.250.135.150",
    "g5": "10.250.135.156",
    "mk22d": "10.250.135.115",
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

PORT_DICT = {"g2": 32, "g3": 33, "g4": 34, "g5": 35}


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
    single_file: Optional[str], input_dirs: Iterable[str]
) -> List[Path]:
    if single_file:
        return [Path(single_file)]

    pdf_paths: List[Path] = []
    for input_dir in input_dirs:
        pdf_paths.extend(get_pdf_files(input_dir))
    return sorted(set(pdf_paths))


def load_markdown(
    path: Path, output_dir: Path, converter: str
) -> Tuple[Optional[str], Optional[Path]]:
    if path.suffix.lower() == ".pdf":
        ensure_directory_exists(str(output_dir))
        md_path = output_dir / f"{path.stem}.md"
        if not md_path.exists():
            md_path = convert_pdf_to_markdown(path, output_dir, converter)
        if md_path and md_path.exists():
            return md_path.read_text(encoding="utf-8"), md_path
        return None, md_path

    if path.suffix.lower() in {".md", ".txt"}:
        return path.read_text(encoding="utf-8"), path

    logger.warning("Skipping unsupported file type: %s", path)
    return None, None


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
    help="Maximum number of characters to send to the model.",
)
@click.option(
    "--limit",
    type=int,
    default=0,
    show_default=True,
    help="Limit number of papers processed (0 means no limit).",
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
    output_dir: str,
    output_jsonl: str,
    converter: str,
    max_chars: int,
    limit: int,
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

    cr = build_client_registry(model, node, port)
    collector = Collector(name="cardiac-filter")

    output_dir_path = Path(output_dir)
    ensure_directory_exists(str(output_dir_path))

    results_path = Path(output_jsonl) if output_jsonl else None
    if results_path:
        ensure_directory_exists(str(results_path.parent))

    paper_paths = collect_input_files(single_file, input_dirs)
    if limit and limit > 0:
        paper_paths = paper_paths[:limit]

    logger.info("Processing %d papers", len(paper_paths))

    for paper_path in paper_paths:
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
