from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

from rfc.formats import Document, RankedDocument, VectorDocument


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
        queries: list[DocQuery | TextQuery | UrlQuery | NameQuery],
        sort: bool = True,
        decay: Callable | None = None,
        skip_visited: bool = False,
    ) -> tuple[list[RankedDocument], list[VectorDocument]]:
        ...

    @abstractmethod
    def to_doc(
        self, query: DocQuery | TextQuery | UrlQuery | NameQuery
    ) -> Document:
        ...

    def query(self, query: str, sort: bool = True) -> list[RankedDocument]:
        return self.query_docs([TextQuery(text=query)], sort=sort)[0]
