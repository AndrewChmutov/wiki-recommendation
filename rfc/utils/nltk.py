import nltk


def _download_if_not_exists(package: str, prefix: str = "") -> None:
    if prefix:
        prefix = prefix + "/"
    try:
        nltk.data.find(prefix + package)
    except LookupError:
        nltk.download(package, quiet=True)


def ensure_nltk() -> None:
    _download_if_not_exists("punkt", prefix="tokenizers")
    _download_if_not_exists("punkt_tab", prefix="tokenizers")
    _download_if_not_exists("wordnet")
