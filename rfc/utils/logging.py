import logging
from collections.abc import Callable
from functools import wraps
from typing import Any

from tqdm.contrib.logging import logging_redirect_tqdm


class CustomFormatter(logging.Formatter):
    grey: str = "\x1b[38;20m"
    green: str = "\x1b[32;20m"
    yellow: str = "\x1b[33;20m"
    red: str = "\x1b[31;20m"
    bold_red: str = "\x1b[31;1m"
    reset: str = "\x1b[0m"
    format_str: str = (
        "%(asctime)s %(name)s(%(levelname)s) - "
        "%(message)s (%(filename)s:%(lineno)d)"
    )

    FORMATS = {
        logging.DEBUG: grey + format_str + reset,
        logging.INFO: green + format_str + reset,
        logging.WARNING: yellow + format_str + reset,
        logging.ERROR: red + format_str + reset,
        logging.CRITICAL: bold_red + format_str + reset
    }

    def format(self, record: logging.LogRecord) -> str:
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def get_logger(
    name: str = "", verbosity: int | str = logging.INFO
) -> logging.Logger:
    verbosity = logging._checkLevel(verbosity)

    logger = logging.getLogger(name)
    logger.setLevel(verbosity)

    ch = logging.StreamHandler()
    ch.setFormatter(CustomFormatter())
    logger.addHandler(ch)

    return logger


def tqdm_logging(logger: logging.Logger) -> Callable:
    def wrapped(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            with logging_redirect_tqdm([logger]):
                return func(*args, **kwargs)
        return wrapper
    return wrapped

