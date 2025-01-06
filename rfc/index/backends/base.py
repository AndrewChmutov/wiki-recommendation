from abc import ABC, abstractmethod

from sklearn.feature_extraction.text import TfidfVectorizer


class BaseTfidfVectorizer(TfidfVectorizer, ABC):
    @abstractmethod
    def _tokenizer(self, texts: list[str]) -> list[list[str]]: ...
