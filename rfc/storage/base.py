from abc import ABC, abstractmethod
from enum import IntFlag, auto

from rfc.formats import Document
from rfc.index.base import Index


class By(IntFlag):
    URL = auto()
    NAME = auto()


class Storage[IndexType: Index](ABC):
    @abstractmethod
    def insert(self, document: Document) -> None: ...

    @abstractmethod
    def load(self) -> IndexType: ...

    @abstractmethod
    def save(self, index: IndexType) -> None: ...

    @abstractmethod
    def clear_docs(self) -> None: ...

    @abstractmethod
    def clear_index(self) -> None: ...

    @abstractmethod
    def fetchone(self, arg: str, by: By) -> Document: ...

    @abstractmethod
    def fetchall(self) -> list[Document]: ...
