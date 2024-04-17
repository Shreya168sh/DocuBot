import structlog
from structlog import BoundLogger, getLogger
from typing import Optional
from pathlib import Path
from datetime import datetime
from config import PathConfigurations


# Create a logger using class and methods
class Logger:
    log: BoundLogger

    def __init__(
            self, 
            name: str,
            file_path: Optional[Path] = None):
        """
        Initializes a new instance of the Logger class.

        Args:
            name (str): The name of the logger.
            file_path (Optional[Path], optional): The file path to log to. Defaults to None.

        Returns:
            None
        """
        if file_path is None:
            file_path = Path(PathConfigurations.LOG_DIR).joinpath(name).joinpath(str(datetime.now().date())).with_suffix(".log")
        self.log = getLogger(name)

    def debug(self, msg, *args, **kwargs):
        """
        Logs a debug message with optional arguments and keyword arguments.

        Args:
            msg (str): The debug message to be logged.
            *args: Optional arguments to be formatted into the message.
            **kwargs: Optional keyword arguments to be formatted into the message.
        """
        self.log.debug(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        """
        Logs an information message with optional arguments and keyword arguments.
        
        Args:
            msg (str): The information message to be logged.
            *args: Optional arguments to be formatted into the message.
            **kwargs: Optional keyword arguments to be formatted into the message.
        """
        self.log.info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        """
        Logs a warning message with optional arguments and keyword arguments.

        Args:
            msg (str): The warning message to be logged.
            *args: Optional arguments to be formatted into the message.
            **kwargs: Optional keyword arguments to be formatted into the message.
        """
        self.log.warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        """
        Logs an error message with optional arguments and keyword arguments.

        Args:
            msg (str): The error message to be logged.
            *args: Optional arguments to be formatted into the message.
            **kwargs: Optional keyword arguments to be formatted into the message.
        """
        self.log.error(msg, *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        """
        Logs a critical message with optional arguments and keyword arguments.

        Args:
            msg (str): The critical message to be logged.
            *args: Optional arguments to be formatted into the message.
            **kwargs: Optional keyword arguments to be formatted into the message.
        """
        self.log.critical(msg, *args, **kwargs)
