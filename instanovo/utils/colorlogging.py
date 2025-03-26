import logging

from rich.console import Console
from rich.logging import RichHandler


class ColorLog:
    """A logging utility class that integrates with Rich for enhanced console output.

    (based on https://stackoverflow.com/a/79225597)
    """

    def __init__(self, console: Console, name: str) -> None:
        message_format = "%(message)s"
        rich_handler = RichHandler(
            console=console
        )  # Log and progress should use the same console instance
        rich_handler.setLevel(logging.INFO)
        rich_handler.setFormatter(logging.Formatter(message_format))

        logging.basicConfig(
            level=logging.NOTSET,
            format=message_format,
            datefmt="[%X]",
            handlers=[rich_handler],
        )

        # Suppress INFO logs from the datasets package
        logging.getLogger("datasets").setLevel(logging.ERROR)

        self.logger = logging.getLogger(name)
