import re
from queue import Queue
from typing import Literal

import bs4
import inscriptis
import requests
from tqdm import trange

from rfc.formats import Document
from rfc.storage.base import Storage
from rfc.utils.logging import Logger


def scrap_single(url: str) -> Document:
    page = requests.get(url)
    html = page.text
    text = inscriptis.get_text(html)
    text = re.sub(r"\n {3,}", "\n", text, flags=re.M)
    text = re.sub(r"\n{3,}", "\n\n", text, flags=re.M)
    return Document(url=url, name="", text=text)


def scrap(
    starting_nodes: list[str],
    limit: int = 50,
    storage: Storage | None = None,
    mode: None | Literal["append"] | Literal["overwrite"] = None,
) -> list[Document]:
    base = "https://datatracker.ietf.org"
    pattern = "https://datatracker.ietf.org/doc/html/rfc"
    starting_nodes = starting_nodes or [
        "https://datatracker.ietf.org/doc/html/rfc791"
    ]
    Logger.info(f"Starting nodes: {starting_nodes}")

    nodes = Queue()
    visited = {}
    skipped = set()

    for node in starting_nodes:
        nodes.put(node)

    gen = trange(limit)
    iter = gen.__iter__()
    docs = []

    match mode:
        case "overwrite":
            assert storage
            Logger.info("Overwriting the documents")
            storage.clear_docs()
            storage.clear_index()
        case "append":
            assert storage
            Logger.info("Appending to documents")

    while nodes and len(visited) < limit:
        # Get from queue
        node = nodes.get()

        # Relative path
        if node.startswith("/") and not node.startswith(base):
            node = base + node

        if (
            node in visited  # visited
            or node in skipped  # skipped
            or "#" in node  # subsection
            or node.startswith("mailto")  # mail
            or not node.startswith(pattern)  # match pattern
        ):
            continue

        Logger.debug(f"Scraping {len(visited) + 1}/{limit} - {node}")

        # Retrieve request
        try:
            page = requests.get(node)
        except Exception as ex:
            skipped.add(node)
            Logger.warning(
                f"Failed to perform a request to {node} due to exception: {ex}"
            )
            continue

        # Skip if error
        if not (200 <= page.status_code < 300):
            skipped.add(node)
            Logger.warning(
                f"Failed to perform a request to {node}, "
                f"status code: {page.status_code}"
            )
            continue

        # Expand neighbors
        html = page.text
        parser = bs4.BeautifulSoup(html, features="html.parser")
        try:
            name = parser.find_all("title", limit=1)[0].string.strip()
        except IndexError:
            skipped.add(node)
            Logger.warning(f"Failed to retrieve name of the  document {node}")
            continue

        # Parse
        for a in parser.find_all("a", href=True):
            nodes.put(a["href"])

        text = html
        text = inscriptis.get_text(html)
        text = re.sub(r"\n {3,}", "\n", text, flags=re.M)
        text = re.sub(r"\n{3,}", "\n\n", text, flags=re.M)

        visited[node] = (name, text)
        next(iter)
        doc = Document(node, name, text)
        docs.append(doc)

        if mode in ["overwrite", "append"]:
            assert storage
            storage.insert(doc)

    next(iter, None)
    gen.close()

    return docs
