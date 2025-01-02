import math
import multiprocessing as mp
from typing import Any

import spacy
from sklearn.feature_extraction.text import TfidfVectorizer

from rfc.index.backends.base import BaseTfidfVectorizer


def download_if_not_exists(name: str) -> None:
    if not spacy.util.is_package(name):
        spacy.cli.download(name)  # pyright: ignore[reportAttributeAccessIssue]


class SpaCyTfidfVectorizer(BaseTfidfVectorizer):
    def __init__(self, spacy_model_name: str, **kwargs) -> None:
        TfidfVectorizer.__init__(self, **kwargs)
        self.spacy_model_name = spacy_model_name
        self.nlp = spacy.load(
            self.spacy_model_name,
            disable=["parser", "ner", "tagger", "attribute_ruler"]
        )

    def _tokenizer(self, texts: list[str]) -> list[list[str]]:
        new_texts = []
        for text in self.nlp.pipe(
            texts,
            n_process=4,
        ):
            new_texts.append([
                token.lemma_
                for token in text
                if not token.is_stop and not token.is_punct
            ])
        return new_texts

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        del state["nlp"]
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self.nlp = spacy.load(self.spacy_model_name)
