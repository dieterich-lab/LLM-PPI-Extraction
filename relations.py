from typing import List, Literal

from pydantic import BaseModel, Field


class Triple(BaseModel):
    head: str = Field(description="head entity of the triple")
    relation: Literal["INTERACTS_WITH"]
    tail: str = Field(description="tail entity name of the triple")
    confidence: Literal["high", "low"] = Field(
        "if this relation was extracted with high confidence or not"
    )


class Triples(BaseModel):
    triples: List[Triple] = Field(
        description="list of triples that describe interactions between two biological entities"
    )
