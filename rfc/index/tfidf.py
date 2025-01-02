from typing import Callable, Self, overload
from uuid import uuid4

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from rfc.formats import Document, RankedDocument
from rfc.index.backends.base import BaseTfidfVectorizer
from rfc.index.base import DocQuery, Index, NameQuery, TextQuery, UrlQuery
from rfc.scrap import scrap_single
from rfc.utils.logging import Logger
from rfc.utils.nltk import ensure_nltk


class TfidfIndex(Index):
    def __init__(
        self,
        tfidf: pd.DataFrame,
        bow: pd.DataFrame,
        idf: dict[str, float],
        docs: list[Document],
        vectorizer: BaseTfidfVectorizer,
    ) -> None:
        ensure_nltk()
        self.tfidf = tfidf
        self.bow = bow
        self.idf = idf
        self.docs = docs
        self.vectorizer = vectorizer

    @classmethod
    def from_docs(
        cls,
        docs: list[Document],
        tfidf_vectorizer: BaseTfidfVectorizer,
    ) -> Self:
        Logger.info("Computing TF-IDF")
        tfidf, bow = cls._vectorize(
            docs, vectorizer=tfidf_vectorizer
        )
        Logger.info("Computing BOW")
        idf = {
            term: val
            for term, val in zip(tfidf.columns, tfidf_vectorizer.idf_)
        }

        return cls(
            tfidf=tfidf,
            bow=bow,
            idf=idf,
            docs=docs,
            vectorizer=tfidf_vectorizer,
        )

    @overload
    @classmethod
    def _vectorize(
        cls,
        docs: list[Document],
        vectorizer: BaseTfidfVectorizer,
        tfidf: pd.DataFrame,
    ) -> pd.DataFrame:
        ...

    @overload
    @classmethod
    def _vectorize(
        cls,
        docs: list[Document],
        vectorizer: BaseTfidfVectorizer,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        ...

    @classmethod
    def _vectorize(
        cls,
        docs: list[Document],
        vectorizer: BaseTfidfVectorizer,
        tfidf: pd.DataFrame | None = None,
    ) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
        if tfidf is None:
            texts = [doc.text for doc in docs]
            Logger.info("Performing lemmatization")
            texts = [" ".join(ts) for ts in vectorizer._tokenizer(texts)]
            vectorizer.fit(texts)
            Logger.info("Fitting and Transforming TF-IDF")
            tfidf_data = vectorizer.transform(texts).toarray()
            tfidf_features = vectorizer.get_feature_names_out()
            tfidf = pd.DataFrame(
                tfidf_data,
                columns=tfidf_features,
                index=[doc.url for doc in docs],  # pyright: ignore[reportArgumentType]
            )

            Logger.info("Fitting and Transforming Bow")
            bow_data = CountVectorizer(vocabulary=tfidf_features).transform(texts).toarray()
            bow_features = vectorizer.get_feature_names_out()
            bow = pd.DataFrame(
                bow_data,
                columns=bow_features,
                index=[doc.url for doc in docs],  # pyright: ignore[reportArgumentType]
            )
            return tfidf, bow
        else:
            urls = [doc.url for doc in docs]
            existing_urls = []
            non_existing_docs = []
            for doc in docs:
                if doc.url in tfidf.index:
                    existing_urls.append(doc.url)
                else:
                    non_existing_docs.append(doc)

            if non_existing_docs:
                texts = [doc.text for doc in non_existing_docs]
                Logger.info("Performing lemmatization")
                texts = [" ".join(ts) for ts in vectorizer._tokenizer(texts)]

                Logger.info("Transforming TF-IDF")
                data = vectorizer.transform(texts).toarray()
                features = vectorizer.get_feature_names_out()
                non_existing_df = pd.DataFrame(
                    data,
                    columns=features,
                    index=[doc.url for doc in non_existing_docs],  # pyright: ignore[reportArgumentType]
                )

                existing_df = tfidf.loc[existing_urls]
                return pd.concat([non_existing_df, existing_df]).loc[urls]
            return tfidf.loc[urls]

    def query_docs(
        self,
        queries: list[DocQuery | TextQuery | UrlQuery],
        sort: bool = True,
        decay: Callable | None = None,
        skip_visited: bool = False
    ) -> list[RankedDocument]:
        assert (
            (len(queries) > 1 and decay is not None) or
            (len(queries) == 1 and decay is None)
        )
        self.check_integrity()
        all_urls = [doc.url for doc in self.docs]

        docs = []
        for query in queries:
            if isinstance(query, DocQuery):
                doc = query.doc
            elif isinstance(query, TextQuery):
                doc = Document(url=str(uuid4()), name="", text=query.text)
            elif isinstance(query, UrlQuery):
                url = query.url
                if url in all_urls:
                    doc = next(filter(lambda x: x.url == url, self.docs))
                else:
                    doc = scrap_single(query.url)
            elif isinstance(query, NameQuery):
                name = query.name
                doc = next(filter(lambda x: name in x.name, self.docs))
            else:
                raise NotImplementedError
            docs.append(doc)

        tfidf = self._vectorize(docs, self.vectorizer, tfidf=self.tfidf)

        query_matrix = tfidf.to_numpy()
        tfidf_matrix = self.tfidf.loc[all_urls].to_numpy()

        similarity_matrix = cosine_similarity(query_matrix, tfidf_matrix)
        similarity_vec = similarity_matrix.mean(0)

        # Decay
        if decay is not None:
            decay_vec = decay(np.arange(len(similarity_vec)))
            similarity_vec = similarity_vec * decay_vec

        ranked_docs = list(map(
            lambda x: RankedDocument.from_document(*x),
            zip(
                self.docs,
                list(tfidf_matrix),
                similarity_vec,
            )
        ))

        if skip_visited:
            new_urls = (
                set(doc.url for doc in ranked_docs) -
                set(doc.url for doc in docs)
            )
            ranked_docs = [doc for doc in ranked_docs if doc.url in new_urls]

        if sort:
            return sorted(ranked_docs, key=lambda x: -x.rank)
        else:
            return ranked_docs

    def check_integrity(self) -> None:
        return self.check_integrity_cls(self.tfidf, self.idf, self.docs)

    @classmethod
    def check_integrity_cls(
        cls,
        tfidf: pd.DataFrame,
        idf: dict[str, float],
        docs: list[Document],
    ) -> None:
        assert tfidf.columns.to_list() == list(idf.keys())
        assert tfidf.index.tolist() == [doc.url for doc in docs]
