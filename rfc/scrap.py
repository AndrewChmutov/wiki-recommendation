import logging
import re
from queue import Queue

import bs4
import inscriptis
import requests
from tqdm import trange

from rfc.datastore import DataStore
from rfc.formats import Document
from rfc.utils.logging import get_logger, tqdm_logging

LOGGER = get_logger()


@tqdm_logging(LOGGER)
def scrap(
    datastore: DataStore | None = None,
    starting_nodes: list[str] | None = None,
    limit: int = 50,
    verbosity: int | str = logging.INFO,
    insert: bool = True,
) -> list[Document]:
    if datastore is None and insert:
        raise ValueError("Datastore is required")

    base = "https://datatracker.ietf.org"
    pattern = "https://datatracker.ietf.org/doc/html/rfc"
    starting_nodes = starting_nodes or ["https://datatracker.ietf.org/doc/html/rfc791"]
    LOGGER.setLevel(verbosity)
    LOGGER.info(f"Starting nodes: {starting_nodes}")

    nodes = Queue()
    visited = {}
    skipped = set()

    for node in starting_nodes:
        nodes.put(node)

    gen = trange(limit)
    iter = gen.__iter__()
    docs = []

    if insert:
        assert datastore
        datastore.clear_docs()

    while nodes and len(visited) < limit:
        # Get from queue
        node = nodes.get()

        # Relative path
        if node.startswith("/") and not node.startswith(base):
            node = base + node

        if (
            node in visited                  # visited
            or node in skipped               # skipped
            or "#" in node                   # subsection
            or node.startswith("mailto")     # mail
            or not node.startswith(pattern)  # match pattern
        ):
            continue

        LOGGER.info(f"Scraping {len(visited) + 1}/{limit} - {node}")

        # Retrieve request
        try:
            page = requests.get(node)
        except Exception as ex:
            skipped.add(node)
            LOGGER.warning(
                f"Failed to perform a request to {node} due to exception: {ex}"
            )
            continue

        # Skip if error
        if not (200 <= page.status_code < 300):
            skipped.add(node)
            LOGGER.warning(
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
            LOGGER.warning(
                f"Failed to retrieve name of the  document {node}"
            )
            continue

        # Parse
        for a in parser.find_all("a", href=True):
            nodes.put(a["href"])

        text = html
        text = inscriptis.get_text(html)
        text = re.sub(r"\n {3,}", "\n", text, flags=re.M)
        text = re.sub(r"\n{3,}", "\n\n", text, flags=re.M)
        text = text.lower()

        visited[node] = (name, text)
        next(iter)
        doc = Document(node, name, text)
        docs.append(doc)

        if insert:
            assert datastore
            datastore.insert_doc(doc)

    next(iter, None)
    gen.close()

    return docs
