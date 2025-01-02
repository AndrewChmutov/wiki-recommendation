import json
import pickle
import sqlite3
from functools import wraps
from pathlib import Path
from typing import Callable, Concatenate, ParamSpec, TypeVar

import pandas as pd

from rfc.formats import Document
from rfc.index.tfidf import TfidfIndex
from rfc.storage.base import By, Storage
from rfc.utils.logging import Logger

P = ParamSpec("P")
R = TypeVar("R")
WrappedMethod = Callable[Concatenate["TfidfStorage", P], R]


class TfidfStorage(Storage):
    IndexType = TfidfIndex

    def __init__(self, db: Path | str, vectorizer_path: Path | str) -> None:
        self.conn = sqlite3.connect(db)
        self.vectorizer_path = Path(vectorizer_path)
        self._create_doc_tables()

    @staticmethod
    def _connection(func: WrappedMethod[P, R]) -> WrappedMethod[P, R]:
        @wraps(func)
        def _wrapper(self, *args: P.args, **kwargs: P.kwargs) -> R:  # noqa: ANN001
            with self.conn:
                output = func(self, *args, **kwargs)
            return output
        return _wrapper

    @_connection
    def insert(self, document: Document) -> None:
        query = """INSERT INTO documents VALUES(?, ?, ?);"""
        self.conn.execute(query, (document.url, document.name, document.text))

    @_connection
    def load(self) -> IndexType:
        Logger.info("Loading the index")
        assert self.vectorizer_path.is_file()
        assert self._check_index_tables()

        docs = self.fetchall()
        urls = [doc.url for doc in docs]
        tfidf = self._load_value_table("tfidf")
        bow = self._load_value_table("bow")
        idf = self._load_idf()
        with self.vectorizer_path.open("rb") as file:
            vectorizer = pickle.load(file)

        return TfidfIndex(
            tfidf=tfidf.loc[urls],
            bow=bow.loc[urls],
            idf=idf,
            docs=docs,
            vectorizer=vectorizer
        )

    @_connection
    def save(self, index: IndexType) -> None:
        Logger.info("Saving the index")
        index.check_integrity()
        self._create_index_tables()

        with self.vectorizer_path.open("wb") as file:
            pickle.dump(index.vectorizer, file)

        # Fill TF-IDF
        query = "INSERT INTO tfidf VALUES(?, ?, ?)"
        for i, row in index.tfidf.iterrows():
            for term, value in row.items():
                self.conn.execute(query, (i, term, value))

        # Fill BOW
        query = "INSERT INTO bow VALUES(?, ?, ?)"
        for i, row in index.bow.iterrows():
            for term, value in row.items():
                self.conn.execute(query, (i, term, value))

        # Fill IDF
        query = "INSERT INTO idf VALUES(?, ?)"
        for item in index.idf.items():
            self.conn.execute(query, item)

    @_connection
    def clear_docs(self) -> None:
        self.conn.execute("DELETE FROM documents;")

    @_connection
    def clear_index(self) -> None:
        self.conn.execute("DROP TABLE IF EXISTS tfidf")
        self.conn.execute("DROP TABLE IF EXISTS bow")
        self.conn.execute("DROP TABLE IF EXISTS idf")

    @_connection
    def fetchone(self, arg: str, by: By) -> Document:
        match by:
            case By.URL:
                query = "SELECT url, name, text FROM documents WHERE url = ?;"
            case By.NAME:
                query = "SELECT url, name, text FROM documents WHERE name = ?;"

        res = self.conn.execute(query, (arg,)).fetchone()
        if res is None:
            raise ValueError(f'Couldn\'t find a document with "{arg}" by {by}')

        url, name, text = res
        return Document(url=url, name=name, text=text)

    @_connection
    def fetchall(self) -> list[Document]:
        query = "SELECT url, name, text FROM documents"
        return [
            Document(url=url, name=name, text=text)
            for url, name, text in self.conn.execute(query).fetchall()
        ]

    def _create_doc_tables(self) -> None:
        queries = [
            """CREATE TABLE IF NOT EXISTS
                documents(url TEXT PRIMARY KEY, name TEXT, text TEXT);""",
            """CREATE UNIQUE INDEX IF NOT EXISTS
                documents_idx ON documents(url);""",
        ]

        for query in queries:
            self.conn.execute(query)

    def _check_doc_table(self) -> bool:
        return self._check_table("documents")

    def _check_table(self, name: str) -> bool:
        fetched = self.conn.execute(
            """SELECT name FROM sqlite_master 
                WHERE type='table' AND name=?;""",
            (name,)
        ).fetchone()
        return fetched is not None

    def _create_index_tables(self) -> None:
        # Drop existing tables
        self.clear_index()

        # Create tables
        query = """CREATE TABLE tfidf(
            doc_url TEXT KEY,
            term TEXT,
            value FLOAT);"""
        self.conn.execute(query)
        query = """CREATE TABLE bow(
            doc_url TEXT KEY,
            term TEXT,
            value FLOAT);"""
        self.conn.execute(query)
        self.conn.execute("CREATE TABLE idf(term TEXT, val FLOAT);")

    def _check_index_tables(self) -> bool:
        return all(map(self._check_table, ["tfidf", "idf"]))

    def _load_value_table(self, table: str) -> pd.DataFrame:
        # Get aggregated data
        Logger.info(f"Reading {table}")
        query = f"""
            SELECT
                doc_url,
                JSON_GROUP_ARRAY(term),
                JSON_GROUP_ARRAY(value)
            FROM {table} GROUP BY doc_url;
        """
        urls, terms, values = zip(*self.conn.execute(query).fetchall())

        # Convert results to the dictionary
        terms = next(map(json.loads, terms))    # (n_keys)
        values = list(map(json.loads, values))  # (n_keys x n_docs)
        data = {
            term: [doc_key_idx_to_value[i] for doc_key_idx_to_value in values]
            for i, term in enumerate(terms)
        }

        return pd.DataFrame(data, index=urls)

    def _load_idf(self) -> dict[str, float]:
        Logger.info("Reading idf")
        idf = self.conn.execute("SELECT term, val FROM idf").fetchall()
        return {term: value for term, value in idf}
