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
    type=int,
    choices=list(range(6)),
)
parser.add_argument("--port", type=int, choices=[34, 35, 36], default=34)
parser.add_argument("--gpu", type=str, choices=["g2", "g3", "g4", "g5"], default="g4")
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
    "--model", choices=["8b", "70b", "405b", "mixtral", "biollm", "nemo"], default="8b"
)
args = parser.parse_args()
