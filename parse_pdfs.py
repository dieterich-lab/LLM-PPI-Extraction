import os
from pathlib import Path

import pymupdf4llm
from marker.config.parser import ConfigParser
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered

input_train_path = "/beegfs/prj/LINDA_LLM/CardioPriorKnowledge/test_ppi_annotations/download_pubmed_papers/PPI_Train"
input_test_path = "/beegfs/prj/LINDA_LLM/CardioPriorKnowledge/test_ppi_annotations/download_pubmed_papers/PPI_Test"

input_paths = [input_train_path, input_test_path]

output_train_path = "/prj/LINDA_LLM/outputs/parsed_papers/regu_train"
output_test_path = "/prj/LINDA_LLM/outputs/parsed_papers/regu_test"

os.makedirs(output_train_path, exist_ok=True)
os.makedirs(output_test_path, exist_ok=True)

output_paths = [output_train_path, output_test_path]

config = {
    "paginate_output": True,
    "extract_images": False,
}
config_parser = ConfigParser(config)

converter = PdfConverter(
    config=config_parser.generate_config_dict(),
    artifact_dict=create_model_dict(),
)

for input_path, output_path in zip(input_paths, output_paths):
    docs = list(Path(input_path).glob("*.pdf"))
    print(len(docs))
    for i, doc in enumerate(docs):
        print(i, doc.stem)
        rendered = converter(str(doc))
        text, _, images = text_from_rendered(rendered)
        (Path(output_path) / f"{doc.stem}.md").write_bytes(text.encode())

        # md_text = pymupdf4llm.to_markdown(doc)
        # (Path(output_path) / f"{doc.stem}.md").write_bytes(md_text.encode())
