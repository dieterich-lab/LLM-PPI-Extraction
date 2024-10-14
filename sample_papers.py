import os
import random
import shutil

random.seed(0)
from pathlib import Path

task = "ppi"

paper_path = Path(f"/home/pwiesenbach/LINDA_LLM/outputs/parsed_papers/{task}")

paths = [x for x in paper_path.iterdir() if x.is_dir()]

for p in paths:
    sample_dir = p / "100samples"
    os.makedirs(sample_dir, exist_ok=True)
    papers = list(p.glob("*.txt")) + list(p.glob("*.md"))
    samples = random.sample(papers, k=100)
    for s in samples:
        shutil.copy(s, sample_dir)
