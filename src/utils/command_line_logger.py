import logging
from typing import Optional

from colorlog import ColoredFormatter

from utils.string import get_random_string


class CommandLineLogger:
    """
    src.utils.command_line_logger.CommandLineLogger class.
    :desc: Utility class to log colorfully messages to console.
    """
    LOG_FORMAT_DEFAULT = "  %(log_color)s%(levelname)-8s%(reset)s | %(log_color)s%(message)s%(reset)s"

    @property
    def log_level(self):
        return self._log_level

    @log_level.setter
    def log_level(self, log_level: str):
        log_level = log_level.upper()
        if self._log_level == log_level:
            return

        # Update internal param
        self._log_level = log_level
        # Set log level
        logging.root.setLevel(log_level)
        self._stream.setLevel(log_level)
        self._logger.setLevel(log_level)

    @property
    def log_format(self):
        return self._log_format

    @log_format.setter
    def log_format(self, log_format: str):
        if self._log_format == log_format:
            return
        # Update private param
        self._log_format = log_format
        # Set log format
        new_formatter = ColoredFormatter(log_format)
        self._formatter = new_formatter
        self._stream.setFormatter(new_formatter)

    @property
    def logger(self):
        return self._logger

    def __init__(self, log_level: str = 'debug', log_format: str = LOG_FORMAT_DEFAULT, name: Optional[str] = None):
        """
        CommandLineLogger constructor.
        :param (str) log_level: Debug Level (one of 'info', 'debug', 'warning', 'error', 'critical')
        :param (str) log_format: Log format or default
        :param (str) name: logger name (enables logs grouping/isolation)
        """
        self._stream = logging.StreamHandler()
        self._logger = logging.getLogger(name if name else get_random_string(length=10))
        self._formatter = ColoredFormatter(log_format)

        self._log_level = None
        self._log_format = None

        self.log_level = log_level
        self.log_format = log_format

        self._logger.addHandler(self._stream)

    def info(self, message: str, *args, **kwargs):
        return self.logger.info(message, *args, **kwargs)

    def debug(self, message: str, *args, **kwargs):
        return self.logger.debug(message, *args, **kwargs)

    def warning(self, message: str, *args, **kwargs):
        return self.logger.warning(message, *args, **kwargs)

    def error(self, message: str, *args, **kwargs):
        return self.logger.error(message, *args, **kwargs)

    def critical(self, message: str, *args, **kwargs):
        return self.logger.critical(message, *args, **kwargs)
