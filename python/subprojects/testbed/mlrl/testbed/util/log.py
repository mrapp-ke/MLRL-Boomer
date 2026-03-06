"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for writing log messages.
"""
import logging

from typing import Optional, override

from rich.console import Console, ConsoleRenderable
from rich.logging import LogRecord, RichHandler
from rich.style import Style
from rich.text import Text

console = Console(soft_wrap=True)


class LogHandler(RichHandler):
    """
    Customizes the appearance of log messages emitted by Python's "logging" module, depending on the log level.
    """

    STYLE_PER_LOG_LEVEL = {
        logging.DEBUG: Style(dim=True),
        logging.WARNING: Style(color='yellow'),
        logging.ERROR: Style(color='red', bold=True),
        logging.CRITICAL: Style(color='red', bold=True),
    }

    PREFIX_PER_LOG_LEVEL = {
        logging.DEBUG: '○',
        logging.WARNING: '⚠',
        logging.ERROR: '✗',
        logging.CRITICAL: '✗',
    }

    def __init__(self):
        super().__init__(show_time=False, show_level=False, show_path=False)

    @staticmethod
    def format_message(message: str, log_level: int) -> str:
        """
        Formats a given log message, depending on a given log level.

        :param message:     The log message to be formatted
        :param log_level:   The log level
        :return:            The formatted message
        """
        prefix = LogHandler.PREFIX_PER_LOG_LEVEL.get(log_level, '')
        return prefix + (' ' if prefix else '') + message

    @staticmethod
    def get_style(log_level: int) -> Optional[Style]:
        """
        Returns the style to be used for a given log level.

        :param log_level:   The log level
        :return:            The style to be used
        """
        return LogHandler.STYLE_PER_LOG_LEVEL.get(log_level)

    @override
    def render_message(self, record: LogRecord, message: str) -> ConsoleRenderable:
        """
        See :func:`rich.logging.RichHandler.render_message`
        """
        log_level = record.levelno
        return Text(self.format_message(message, log_level), style=self.get_style(log_level))


class Log:
    """
    Allows to write log messages.
    """

    @staticmethod
    def error(message: str, *args, error: Optional[Exception] = None):
        """
        Writes a log message at level `Log.Level.ERROR` and terminates the build system.

        :param message: The log message to be written
        :param args:    Optional arguments to be included in the log message
        :param error:   An optional error to be included in the log message
        """
        log_level = logging.ERROR
        console.print(LogHandler.format_message(message.format(*args), log_level),
                      style=LogHandler.get_style(log_level))

        if error:
            console.print_exception(extra_lines=2)

    @staticmethod
    def warning(message: str, *args):
        """
        Writes a log message at level `Log.Level.WARNING`.

        :param message: The log message to be written
        :param args:    Optional arguments to be included in the log message
        """
        log_level = logging.WARNING
        console.print(LogHandler.format_message(message.format(*args), log_level),
                      style=LogHandler.get_style(log_level))

    @staticmethod
    def success(message: str, *args):
        """
        Writes a log message at level `Log.Level.INFO` indicating successful operation of an operation.

        :param message: The log message to be written
        :param args:    Optional arguments to be included in the log message
        """
        console.print('✓ ' + message.format(*args), style=Style(color='green', bold=True))

    @staticmethod
    def info(message: str, *args):
        """
        Writes a log message at level `Log.Level.INFO`.

        :param message: The log message to be written
        :param args:    Optional arguments to be included in the log message
        """
        log_level = logging.INFO
        console.print(LogHandler.format_message(message.format(*args), log_level),
                      style=LogHandler.get_style(log_level))

    @staticmethod
    def verbose(message: str, *args):
        """
        Writes a log message at level `Log.Level.VERBOSE`.

        :param message: The log message to be written
        :param args:    Optional arguments to be included in the log message
        """
        log_level = logging.DEBUG
        console.print(LogHandler.format_message(message.format(*args), log_level),
                      style=LogHandler.get_style(log_level))
