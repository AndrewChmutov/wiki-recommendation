from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class Document:
    url: str
    name: str
    text: str

    def __str__(self) -> str:
        return f'{self.__class__.__name__}(url="{self.url}", name="{self.name}", text=...)'  # noqa: E501

    def __repr__(self) -> str:
        return self.__str__()


@dataclass
class VectorDocument(Document):
    vector: np.ndarray

    @classmethod
    def from_document(
        cls, doc: Document, vector: np.ndarray
    ):
        return cls(**(
            asdict(doc) | {"vector": vector}
        ))


@dataclass
class RankedDocument(VectorDocument):
    vector: np.ndarray
    rank: float = 0.0

    def __str__(self) -> str:
        return f"{self.rank:.3f} {super().__str__()}"  # noqa: E501

    @classmethod
    def from_document(
        cls, doc: Document, vector: np.ndarray, rank: float
    ):
        return cls(**(
            asdict(doc) | {"vector": vector, "rank": rank}
        ))

