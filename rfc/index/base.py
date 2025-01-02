from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

from rfc.formats import Document, RankedDocument


@dataclass
class DocQuery:
    doc: Document


@dataclass
class TextQuery:
    text: str


@dataclass
class UrlQuery:
    url: str


@dataclass
class NameQuery:
    name: str


class Index(ABC):
    @abstractmethod
    def query_docs(
        self,
        queries: list[DocQuery | TextQuery | UrlQuery],
        sort: bool = True,
        decay: Callable | None = None,
        skip_visited: bool = False,
    ) -> list[RankedDocument]:
        ...

    def query(self, query: str, sort: bool = True) -> list[RankedDocument]:
        return self.query_docs([TextQuery(text=query)], sort=sort)
