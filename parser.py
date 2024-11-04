import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--target",
    nargs="?",
    const="tf",
    type=str,
    default="tf",
    choices=["tf", "ppi", "both"],
)
parser.add_argument(
    "--style",
    nargs="?",
    const="1",
    type=str,
    default="1",
    choices=list(map(str, (range(6)))),
)
parser.add_argument("--port", type=int, choices=[34, 35, 36])
parser.add_argument("--gpu", type=int, choices=["g2", "g3", "g4", "g5"])
parser.add_argument(
    "--parser",
    nargs="?",
    const="llama_parse",
    type=str,
    default="llama_parse",
    choices=["llama_parse", "marker"],
)
parser.add_argument(
    "--simple",
    action="store_true",
)
parser.add_argument(
    "--curated",
    action="store_true",
)
parser.add_argument(
    "--dev",
    action="store_true",
)
parser.add_argument(
    "--model", choices=["8b", "70b", "405b", "mixtral", "biollm"], default="8b"
)
args = parser.parse_args()
