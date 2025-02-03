from examples import (
    LrExamples,
    ProteinExamples,
    ProteinNerExamples,
    TfGeneExamples,
    TfGeneNerExamples,
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
