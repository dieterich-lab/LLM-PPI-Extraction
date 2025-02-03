from enum import Enum
from typing import List, Literal

from pydantic import BaseModel, Field


class ProteinTriple(BaseModel):
    head: str = Field(description="Head protein entity.")
    relation: Literal["INTERACTS_WITH"] = Field(description="Protein-protein relation.")
    tail: str = Field(description="Tail protein entity.")


class ProteinTriples(BaseModel):
    """
    A class that contains a list of protein relations in the form of triples.
    """

    triples: List[ProteinTriple] = Field(description="List of all extracted Triples.")


class TFGeneTriple(BaseModel):
    head: str = Field(description="Head gene entity")
    relation: Literal["REGULATES"] = Field(
        description="Transcription factor-gene relation."
    )
    tail: str = Field(description="Tail gene entity.")


class TFGeneTriples(BaseModel):
    """
    A class that contains a list of transcription factor/gene relations in the form of triples.
    """

    triples: List[TFGeneTriple] = Field(description="List of all extracted Triples.")


class Proteins(BaseModel):
    """
    A class that contains a list of proteins.
    """

    entities: List[str] = Field(description="List of all extracted proteins.")


class GenesAndTranscriptionFactors(BaseModel):
    """
    A class that contains a list of genes and transcription factors.
    """

    entities: List[str] = Field(
        description="List of all extracted genes and transcription factors."
    )
