from examples import (
    LrExamples,
    ProteinExamples,
    ProteinNerExamples,
    TfGeneExamples,
    TfGeneNerExamples,
)
from structured_classes import (
    LR_Triples_Simple,
    PPI_Triples_Simple,
    TFGeneTriples,
    Triples_Simple,
)

example_dict = {
    "ppi": ProteinExamples,
    "tf": TfGeneExamples,
    "lr": LrExamples,
    "both": ProteinExamples + TfGeneExamples,
}
ner_example_dict = {
    "ppi": ProteinNerExamples,
    "tf": TfGeneNerExamples,
}

schema_dict = {
    "ppi": PPI_Triples_Simple,
    "tf": TFGeneTriples,
    "lr": LR_Triples_Simple,
    "both": Triples_Simple,
}
