import glob
import logging
import subprocess
from pathlib import Path

logging.basicConfig(
    filename="/beegfs/prj/LINDA_LLM/outputs/logs/llama_parse.log",
    filemode="a+",
    format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
    datefmt="%d,%H:%M",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# task = "tf"
task = "ppi"

path_dict = {
    "ppi": [
        "/beegfs/prj/LINDA_LLM/PubMed_Resources/Papers_Human_Cardiac_Alternative_Splicing/pdf_separate",
        "/beegfs/prj/LINDA_LLM/PubMed_Resources/Papers_Human_Cardiac_Signaling/pdf_separate",
    ],
    "tf": ["/beegfs/prj/LINDA_LLM/PubMed_Resources/Papers_Human_TF_Genes/pdf_separate"],
}

paths = path_dict[task]
_raw_docs = [glob.glob(path + "/*.pdf") for path in paths]
raw_docs = [y for x in _raw_docs for y in x]
print(len(raw_docs))

parsed_papers_path = f"/beegfs/prj/LINDA_LLM/outputs/parsed_papers/{task}/marker/"

# paths = ["/home/pwiesenbach/LINDA_LLM/test"]
for path in paths:
    print(path)
    result = subprocess.call(
        ["marker", path, parsed_papers_path, "--workers", "4", "--min_length", "1000"],
    )

logger.info(result)
