import logging
import os
from typing import Literal

from rich.console import Console
from rich.logging import RichHandler

from instanovo.__init__ import get_rank
from instanovo.constants import LOGGING_SHOW_PATH, LOGGING_SHOW_TIME, USE_RICH_HANDLER


class DynamicRankFormatter(logging.Formatter):
    """A formatter that dynamically includes rank information in the logger name."""

    def __init__(
        self,
        fmt: str | None = None,
        datefmt: str | None = None,
        style: Literal["%", "{", "$"] = "%",
    ) -> None:
        super().__init__(fmt, datefmt, style)

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record.

        Args:
            record (logging.LogRecord): The log record to format.

        Returns:
            str: The formatted log record
        """
        rank = get_rank()

        # If we have a rank, append it to the logger name
        if rank is not None:
            record.msg = f"[RANK {rank}] {record.msg}"

        return super().format(record)


class ColorLog:
    """A logging utility class that integrates with Rich for enhanced console output.

    (based on https://stackoverflow.com/a/79225597)
    """

    def __init__(self, console: Console, name: str) -> None:
        message_format = "%(message)s"  # Include logger name in format

        aichor_enabled = "AICHOR_LOGS_PATH" in os.environ

        rich_handler = RichHandler(
            console=console,
            show_time=LOGGING_SHOW_TIME and not aichor_enabled,
            show_path=LOGGING_SHOW_PATH and not aichor_enabled,
        )
        rich_handler.setLevel(logging.INFO)
        rich_handler.setFormatter(DynamicRankFormatter(message_format))

        if USE_RICH_HANDLER:
            logging.basicConfig(
                level=logging.NOTSET,
                format=message_format,
                datefmt="[%X]",
                handlers=[rich_handler],
            )
        else:
            logging.basicConfig(
                level=logging.INFO,
                format=message_format,
            )

        # Suppress INFO logs from the datasets package
        logging.getLogger("datasets").setLevel(logging.ERROR)

        self.logger = logging.getLogger(name)
