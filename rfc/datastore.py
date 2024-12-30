import json
import sqlite3
from collections.abc import Callable
from enum import IntFlag, auto
from functools import wraps
from pathlib import Path
from typing import (
    Concatenate,
    ParamSpec,
    TypeVar,
)

import pandas as pd

from rfc.formats import Document
from rfc.utils.logging import get_logger

P = ParamSpec("P")
R = TypeVar("R")
WrappedMethod = Callable[Concatenate["DataStore", P], R]
LOGGER = get_logger(Path(__file__).name)


class By(IntFlag):
    ANY = auto()
    URL = auto()
    NAME = auto()


class DataStore:
    @staticmethod
    def connection(func: WrappedMethod[P, R]) -> WrappedMethod[P, R]:
        @wraps(func)
        def _wrapper(self, *args: P.args, **kwargs: P.kwargs) -> R:  # noqa: ANN001
            with self.conn:
                output = func(self, *args, **kwargs)
            return output
        return _wrapper

    def __init__(self, location: str) -> None:
        self.conn = sqlite3.connect(location)
        self._create_doc_table()

    @connection
    def _create_doc_table(self) -> None:
        queries = [
            """CREATE TABLE IF NOT EXISTS
                documents(url TEXT PRIMARY KEY, name TEXT, text TEXT);""",
            """CREATE UNIQUE INDEX IF NOT EXISTS
                documents_idx ON documents(url);""",
        ]

        for query in queries:
            self.conn.execute(query)

    @connection
    def insert_doc(self, document: Document) -> None:
        query = """INSERT INTO documents VALUES(?, ?, ?);"""
        self.conn.execute(query, (document.url, document.name, document.text))

    @connection
    def clear_docs(self) -> None:
        self.conn.execute("DELETE FROM documents;")

    @connection
    def create_index(
        self,
        tfidf: pd.DataFrame,
        tf: pd.DataFrame,
        idf: dict[str, float]
    ) -> None:
        if list(tfidf.columns) != list(idf.keys()):
            raise ValueError("Terms in columns do not align with idf keys")

        # Drop existing tables
        self.conn.execute("DROP TABLE IF EXISTS tfidf")
        self.conn.execute("DROP TABLE IF EXISTS tf")
        self.conn.execute("DROP TABLE IF EXISTS idf")

        # Create tables
        query = """CREATE TABLE tfidf(
            doc_url TEXT KEY,
            term TEXT,
            value FLOAT);"""
        self.conn.execute(query)
        query = """CREATE TABLE tf(
            doc_url TEXT KEY,
            term TEXT,
            value INTEGER);"""
        self.conn.execute(query)
        self.conn.execute("CREATE TABLE idf(term TEXT, val FLOAT);")

        # Fill tables
        tables = ["tfidf", "tf"]
        contents = [tfidf, tf]
        for table, content in zip(tables, contents):
            query = f"INSERT INTO {table} VALUES(?, ?, ?)"
            insert_args = []
            for index, row in content.iterrows():
                for term, value in row.items():
                    insert_args.append((index, term, value))
            self.conn.executemany(query, insert_args)

        query = "INSERT INTO idf VALUES(?, ?)"
        self.conn.executemany(query, idf.items())

    @connection
    def get_doc(
        self, *args: tuple[str], by: By = By.ANY
    ) -> Document:
        match by:
            case By.URL:
                query = "SELECT url, name, text FROM documents WHERE url = ?;"
            case By.NAME:
                query = "SELECT url, name, text FROM documents WHERE name = ?;"
            case _:
                raise ValueError(f"Can't query by {by}")

        res = self.conn.execute(query, args).fetchone()
        if res is None:
            raise ValueError(f"Couldn't find a document by {args}")

        url, name, text = res
        return Document(url=url, name=name, text=text)

    @connection
    def get_docs(
        self, *args: tuple[str], by: By = By.ANY
    ) -> list[Document]:
        match by:
            case By.ANY:
                query = "SELECT url, name, text FROM documents"
                return [
                    Document(url=url, name=name, text=text)
                    for url, name, text in self.conn.execute(query).fetchall()
                ]
            case _:
                return [self.get_doc(*args, by=by)]

    def _load_value_table(self, table: str) -> pd.DataFrame:
        # Get aggregated data
        LOGGER.info(f"Reading {table}")
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
        LOGGER.info("Reading idf")
        idf = self.conn.execute("SELECT term, val FROM idf").fetchall()
        return {term: value for term, value in idf}

    @connection
    def load_index(
        self
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
        tfidf = self._load_value_table("tfidf")
        tf = self._load_value_table("tf")
        idf = self._load_idf()
        assert tfidf.columns.to_list() == tf.columns.to_list()
        assert tfidf.columns.to_list() == list(idf.keys())
        return tfidf, tf, idf
