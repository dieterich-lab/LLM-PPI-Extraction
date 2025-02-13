import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    choices=["llama31", "llama33", "deepseek8b", "deepseek70b"],
    default="deepseek8b",
    help="Alias pointing back to model names of the local Ollama server or the provider.",
)
parser.add_argument(
    "--extractionmode",
    type=str,
    choices=[
        "direct",
        "nerrel",
        # "neronly",
        # "relgiventrueners",
        # "relgivenallners",
    ],
    # default="direct",
    default="nerrel",
)
parser.add_argument(
    "--chattype",
    type=str,
    choices=["oneshot", "stepwise"],
    default="stepwise",
    # default="oneshot",
)
parser.add_argument(
    "--data",
    type=str,
    choices=[
        "ours",
        "5curated",
        "eval",
        "biored",
        "regulatome",
    ],
    default="5curated",
    help="Which data to extract from.",
)
parser.add_argument(
    "--target",
    type=str,
    choices=["ppi", "tf", "lr"],
    default="ppi",
    help="Which entity type you want to extract.",
)
parser.add_argument(
    "--doclevel",
    type=str,
    choices=["chunks", "docs"],
    default="docs",
)
# parser.add_argument(
#     "--style",
#     type=int,
#     choices=list(range(1, 7)),
#     default=1,
#     help="Declare which of our predefined styles you want to chose (see `style_dict` in 'style_templates.py' for the individual prompts.)",
# )
parser.add_argument(
    "--port",
    type=int,
    choices=[34, 35, 36],
    default=34,
    help="Port where the local Ollama server is running.",
)
parser.add_argument(
    "--node",
    type=str,
    choices=["g2", "g3", "g4", "g5", "mk22d"],
    default="g4",
    help="Node alias that defines the ip where the Ollama server is running (see 'llm.py').",
)
parser.add_argument(
    "--parser",
    nargs="?",
    const="llama_parse",
    type=str,
    default="llama_parse",
    choices=["llama_parse", "marker"],
    help="These are aliases pointing back to the folder of parsed PDF files (paths configured in 'get_documents.py' and 'paths.py')",
)
parser.add_argument(
    "--loglevel",
    choices=["error", "warn", "info", "debug", "trace", "off"],
    default="off",
    help="These are aliases pointing back to the folder of parsed PDF files (paths configured in 'get_documents.py' and 'paths.py')",
)
parser.add_argument(
    "--startfromdoc",
    nargs="?",
    const=0,
    type=int,
    default=0,
    help="To exclude the first n documents/chunks from the extraction.",
)
parser.add_argument(
    "--untildoc",
    nargs="?",
    const=0,
    type=int,
    default=0,
    help="To only parse untnil the nth document/chunk.",
)
parser.add_argument(
    "--printpaperpaths",
    action="store_true",
    help="Debugging option to print the paper paths while processing.",
)
parser.add_argument(
    "--nebius",
    action="store_true",
    help="We used Nebius (nebius.com) as provider to run external computations. This changes the Chat Wrapper API (see 'llm.py').",
)
parser.add_argument(
    "--dev",
    action="store_true",
    help="A developing options that stops the scripts from actually saving/overwriting results.",
)
parser.add_argument(
    "--filelist",
    action="store_true",
    default=True,
    help="If true, we append the filename of the current processed file to the list of entities that are extracted. So be careful when ever analysing the 'ner.json' in the `graphdoc_pkl_path` that the last element in a list will then be the filenmae",
)
parser.add_argument(
    "--noexamples",
    action="store_true",
    help="Not yet implemented. Option for future experiments without giving examples to the model (exemplifying zero-shot inference.)",
)
parser.add_argument(
    "--apikey",
    type=str,
    default="NEBIUS_API_KEY_PRP",
    help="If you use an external model provider, this is the API key that is used for it and read out from `os.env`.",
)

args = parser.parse_args()
