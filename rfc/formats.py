from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Self

import numpy as np


@dataclass
class Document:
    url: str
    name: str
    text: str

    def __str__(self) -> str:
        return f"{self.name}: {self.url}"

    def __repr__(self) -> str:
        return self.__str__()

    def with_vec(self, vec: np.ndarray) -> VectorDocument:
        return VectorDocument(**asdict(self), vec=vec)


@dataclass
class VectorDocument(Document):
    vec: np.ndarray

    @classmethod
    def from_document(
        cls, doc: Document, vector: np.ndarray, **kwargs
    ) -> Self:
        return cls(**(
            asdict(doc) | {"vector": vector}
        ))

    def with_rank(
        self, rank: float, similarities: np.ndarray
    ) -> RankedDocument:
        return RankedDocument(
            **asdict(self), similarities=similarities, rank=rank
        )


@dataclass
class RankedDocument(VectorDocument):
    # Similarities to each document in a query
    similarities: np.ndarray
    rank: float = 0.0

    def __str__(self) -> str:
        return f"{self.rank:.3f} {super().__str__()}"  # noqa: E501
