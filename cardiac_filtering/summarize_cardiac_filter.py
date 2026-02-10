import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import click

DEFAULT_RESULTS_PATH = "/prj/LINDA_LLM/outputs/cardiac_filter/results.jsonl"

logger = logging.getLogger("cardiac_filter_summary")


@dataclass
class Metrics:
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0

    def add(self, y_true: bool, y_pred: bool) -> None:
        if y_true and y_pred:
            self.tp += 1
        elif not y_true and y_pred:
            self.fp += 1
        elif not y_true and not y_pred:
            self.tn += 1
        else:
            self.fn += 1

    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 0.0

    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0

    def f1(self) -> float:
        p = self.precision()
        r = self.recall()
        return 2 * p * r / (p + r) if (p + r) else 0.0


def load_results(path: Path) -> Iterable[Dict[str, object]]:
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def normalize_ground_truth(value: Optional[str]) -> Optional[bool]:
    if value is None:
        return None
    value = value.strip().lower()
    if value == "cardiac":
        return True
    if value == "non-cardiac":
        return False
    return None


def summarize_metrics(
    rows: Iterable[Dict[str, object]], positive_label: str
) -> Dict[str, Metrics]:
    metrics_by_key: Dict[str, Metrics] = {
        "ensemble": Metrics(),
        "prompt_1": Metrics(),
        "prompt_2": Metrics(),
        "prompt_3": Metrics(),
    }

    for row in rows:
        ground_truth = normalize_ground_truth(row.get("ground_truth"))
        if ground_truth is None:
            continue

        ensemble_pred = row.get("is_cardiac")
        if isinstance(ensemble_pred, bool):
            metrics_by_key["ensemble"].add(ground_truth, ensemble_pred)

        votes = row.get("votes", [])
        if isinstance(votes, list):
            for vote in votes:
                if not isinstance(vote, dict):
                    continue
                prompt = vote.get("prompt")
                is_cardiac = vote.get("is_cardiac")
                if prompt in metrics_by_key and isinstance(is_cardiac, bool):
                    metrics_by_key[prompt].add(ground_truth, is_cardiac)

    return metrics_by_key


def format_metrics(name: str, metrics: Metrics) -> str:
    return (
        f"{name:10s} | "
        f"tp={metrics.tp:3d} fp={metrics.fp:3d} "
        f"tn={metrics.tn:3d} fn={metrics.fn:3d} | "
        f"precision={metrics.precision():.3f} "
        f"recall={metrics.recall():.3f} "
        f"f1={metrics.f1():.3f}"
    )


@click.command()
@click.option(
    "--results-jsonl",
    default=DEFAULT_RESULTS_PATH,
    show_default=True,
    help="Path to the cardiac filter results JSONL file.",
)
@click.option(
    "--positive-label",
    default="cardiac",
    show_default=True,
    help="Label treated as the positive class.",
)
@click.option("--verbose", is_flag=True, help="Enable debug logging.")
def main(results_jsonl: str, positive_label: str, verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    results_path = Path(results_jsonl)
    if not results_path.exists():
        raise click.ClickException(f"Results file not found: {results_path}")

    rows = list(load_results(results_path))
    if not rows:
        logger.warning("No rows found in %s", results_path)
        return

    metrics_by_key = summarize_metrics(rows, positive_label)

    print("Cardiac filter metrics (positive = cardiac)")
    for name in ["ensemble", "prompt_1", "prompt_2", "prompt_3"]:
        print(format_metrics(name, metrics_by_key[name]))


if __name__ == "__main__":
    main()
