import structlog
from structlog import BoundLogger, getLogger
from typing import Optional
from pathlib import Path
from datetime import datetime
from config import PathConfigurations


# Create a logger using class and methods
class Logger:
    """
    Create a logger for application
    """
    log: BoundLogger


    def __init__(
            self, 
            name: str,
            file_path: Optional[Path] = None):
        if file_path is None:
            file_path = Path(PathConfigurations.LOG_DIR).joinpath(name).joinpath(str(datetime.now().date())).with_suffix(".log")
        self.log = getLogger(name)

    def debug(self, msg, *args, **kwargs):
        self.log.debug(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        self.log.info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self.log.warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self.log.error(msg, *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        self.log.critical(msg, *args, **kwargs)


if __name__ == "__main__":
    print(Logger("test").info("Test"))