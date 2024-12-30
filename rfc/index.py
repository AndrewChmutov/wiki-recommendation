import pickle
from collections.abc import Iterable
from os import PathLike
from pathlib import Path
from typing import Literal

import pandas as pd
from nltk import WordNetLemmatizer, word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from rfc.datastore import DataStore
from rfc.formats import Document, RankedDocument
from rfc.utils.logging import get_logger
from rfc.utils.nltk import ensure_nltk

LOGGER = get_logger(Path(__file__).name)

ensure_nltk()
LEMMATIZER = WordNetLemmatizer()
VECTORIZER_PATH = Path("vectorizer.pkl")


class Index:

    def __init__(
        self,
        datastore: DataStore | None = None,
        vectorizer_path: Path | str = "vectorizer.pkl",
        mode: Literal["create"] | Literal["load"] = "create"
    ) -> None:

        if datastore is None:
            raise ValueError("Datastore is needed")

        self.datastore = datastore

        self.tfidf = None
        self.tf = None
        self.idf = None

        match mode:
            case "create":
                self._create_and_save()
            case "load":
                self._load()
            case _:
                raise ValueError(f"Wrong mode: {mode}")

        self._vectorizer_instance = None
        if (path := Path(vectorizer_path)).is_file():
            with open(path, "rb") as file:
                self._vectorizer_instance = pickle.load(file)

    def _create_and_save(self) -> None:
        # Create from datastore
        docs = self.datastore.get_docs()

        # Admissible computational overhead
        LOGGER.info("Computing BOW")
        tf, _ = self._vectorize(docs, vectorizer_type=CountVectorizer)

        LOGGER.info("Computing TF-IDF")
        tfidf, idf = self._vectorize(docs, save_path=VECTORIZER_PATH)
        assert idf is not None
        assert tfidf.columns.to_list() == tf.columns.to_list()
        assert tfidf.columns.to_list() == list(idf.keys())

        LOGGER.info("Creating database")
        self.datastore.create_index(tfidf, tf, idf)
        self.tfidf = tfidf
        self.tf = tf
        self.idf = idf

    def _load(self) -> None:
        tfidf, tf, idf = self.datastore.load_index()
        self.tfidf = tfidf
        self.tf = tf
        self.idf = idf

    def _vectorize(
        self,
        docs: list[Document],
        vectorizer_type: type[CountVectorizer] = TfidfVectorizer,
        save_path: PathLike | str | None = None
    ) -> tuple[pd.DataFrame, dict[str, float] | None]:
        urls = [doc.url for doc in docs]

        if self.tfidf is not None:
            print("use saved")
            vectorizer_instance = self._vectorizer_instance
        else:
            vectorizer_instance = vectorizer_type(
                tokenizer=self._tokenizer,
                preprocessor=self._preprocessor,
                token_pattern=None,
            )
            vectorizer_instance = vectorizer_instance.fit(docs)

        if save_path is not None:
            LOGGER.info(
                f"Saving {vectorizer_type.__name__} to {save_path}"
            )
            with open(save_path, "wb") as file:
                pickle.dump(vectorizer_instance, file)
        data = vectorizer_instance.transform(docs).toarray()
        features = vectorizer_instance.get_feature_names_out()

        df = pd.DataFrame(data, columns=features, index=urls)

        idf = None
        if isinstance(vectorizer_instance, TfidfVectorizer):
            idf = {
                term: val
                for term, val in zip(features, vectorizer_instance.idf_)
            }

        return df, idf

    @staticmethod
    def _preprocessor(doc: Document) -> str:
        return doc.text

    @staticmethod
    def _tokenizer(text: str) -> Iterable[str]:
        yield from map(LEMMATIZER.lemmatize, word_tokenize(text))

    def recommend_query(self, query: str) -> list[RankedDocument]:
        assert self.tfidf is not None

        document = Document(url="", name="query", text=query.lower())
        tfidf, _ = self._vectorize([document])

        query_matrix = tfidf.to_numpy()
        urls = list(doc.url for doc in self.datastore.get_docs())
        tfidf_matrix = self.tfidf.loc[urls].to_numpy()
        similarity_matrix = cosine_similarity(query_matrix, tfidf_matrix)
        docs = [
            doc.with_rank(similarity)
            for similarity, doc in zip(
                similarity_matrix[0],
                self.datastore.get_docs()
            )
        ]

        return sorted(docs, key=lambda x: -x.rank)

    def recommend(self, history: list[Document]) -> list[RankedDocument]:
        pass
