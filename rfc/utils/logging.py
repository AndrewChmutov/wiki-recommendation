import logging


class CustomFormatter(logging.Formatter):
    grey: str = "\x1b[38;20m"
    green: str = "\x1b[32;20m"
    yellow: str = "\x1b[33;20m"
    red: str = "\x1b[31;20m"
    bold_red: str = "\x1b[31;1m"
    reset: str = "\x1b[0m"
    format_str: str = (
        "%(name)s(%(levelname)s) - "
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


Logger = logging.getLogger()
_handler = logging.StreamHandler()
_handler.setFormatter(CustomFormatter())
Logger.addHandler(_handler)
