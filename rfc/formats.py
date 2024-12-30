from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class Document:
    url: str
    name: str
    text: str

    def with_rank(self, rank: float) -> RankedDocument:
        return RankedDocument(**(asdict(self) | {"rank": rank}))

    def __str__(self) -> str:
        return f'{self.__class__.__name__}(url="{self.url}", name="{self.name}", text=...)'  # noqa: E501


@dataclass
class RankedDocument(Document):
    rank: float = 0.0

    def __str__(self) -> str:
        return f'{self.__class__.__name__}(url="{self.url}", name="{self.name}", rank={self.rank}, text=...)'  # noqa: E501
