"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for writing log messages.
"""
import contextlib
import logging
import os

from contextlib import contextmanager
from typing import Optional, override

from rich.console import Console, ConsoleRenderable
from rich.logging import LogRecord, RichHandler
from rich.panel import Panel
from rich.style import Style
from rich.syntax import Syntax
from rich.text import Text

from mlrl.testbed.util.io import ENCODING_UTF8

console = Console(soft_wrap=True)


@contextmanager
def disable_log():
    """
    Prevents any output from being written to stdout or stderr.
    """
    with open(os.devnull, mode='w', encoding=ENCODING_UTF8) as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield


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
        formatted_message = self.format_message(message, log_level)
        style = self.get_style(log_level)
        return Text(formatted_message, style=style) if style else Text(formatted_message)


class Log:
    """
    Allows to write log messages.
    """

    @staticmethod
    def error(message: str,
              *args,
              error: Optional[Exception] = None,
              box: bool = False,
              box_title: Optional[str] = None):
        """
        Writes a log message at level `Log.Level.ERROR` and terminates the build system.

        :param message:     The log message to be written
        :param args:        Optional arguments to be included in the log message
        :param error:       An optional error to be included in the log message
        :param box:         True, if a box should be surrounded by a box, False otherwise
        :param box_title:   An optional title to be printed at the top of the box surrounding the log message
        """
        log_level = logging.ERROR

        if logging.getLogger().isEnabledFor(log_level):
            formatted_message = LogHandler.format_message(message.format(*args), log_level)
            style = LogHandler.get_style(log_level)

            if box:
                text = Text(formatted_message, style=style) if style else Text(formatted_message)
                console.print(Panel.fit(text, title=box_title))
            else:
                console.print(formatted_message, style=style)

            if error:
                console.print_exception(extra_lines=2)

    @staticmethod
    def warning(message: str, *args, box: bool = False, box_title: Optional[str] = None):
        """
        Writes a log message at level `Log.Level.WARNING`.

        :param message:     The log message to be written
        :param args:        Optional arguments to be included in the log message
        :param box:         True, if a box should be surrounded by a box, False otherwise
        :param box_title:   An optional title to be printed at the top of the box surrounding the log message
        """
        log_level = logging.WARNING

        if logging.getLogger().isEnabledFor(log_level):
            formatted_message = LogHandler.format_message(message.format(*args), log_level)
            style = LogHandler.get_style(log_level)

            if box:
                text = Text(formatted_message, style=style) if style else Text(formatted_message)
                console.print(Panel.fit(text, title=box_title))
            else:
                console.print(formatted_message, style=style)

    @staticmethod
    def success(message: str, *args, box: bool = False, box_title: Optional[str] = None):
        """
        Writes a log message at level `Log.Level.INFO` indicating successful operation of an operation.

        :param message:     The log message to be written
        :param args:        Optional arguments to be included in the log message
        :param box:         True, if a box should be surrounded by a box, False otherwise
        :param box_title:   An optional title to be printed at the top of the box surrounding the log message
        """
        log_level = logging.INFO

        if logging.getLogger().isEnabledFor(log_level):
            formatted_message = '✓ ' + message.format(*args)
            style = Style(color='green', bold=True)

            if box:
                text = Text(formatted_message, style=style) if style else Text(formatted_message)
                console.print(Panel.fit(text, title=box_title))
            else:
                console.print(formatted_message, style=style)

    @staticmethod
    def info(message: str, *args, box: bool = False, box_title: Optional[str] = None):
        """
        Writes a log message at level `Log.Level.INFO`.

        :param message:     The log message to be written
        :param args:        Optional arguments to be included in the log message
        :param box:         True, if a box should be surrounded by a box, False otherwise
        :param box_title:   An optional title to be printed at the top of the box surrounding the log message
        """
        log_level = logging.INFO

        if logging.getLogger().isEnabledFor(log_level):
            formatted_message = LogHandler.format_message(message.format(*args), log_level)
            style = LogHandler.get_style(log_level)

            if box:
                text = Text(formatted_message, style=style) if style else Text(formatted_message)
                console.print(Panel.fit(text, title=box_title))
            else:
                console.print(formatted_message, style=style)

    @staticmethod
    def source_code(source_code: str, *args, language: str, box: bool = False, box_title: Optional[str] = None):
        """
        Writes a log message containing source code in a specific language at level `Log.Level.INFO`.

        :param source_code: The source code to be written
        :param args:        Optional arguments to be included in the log message
        :param language:    The language used by the source code
        :param box:         True, if a box should be surrounded by a box, False otherwise
        :param box_title:   An optional title to be printed at the top of the box surrounding the log message
        """
        log_level = logging.INFO

        if logging.getLogger().isEnabledFor(log_level):
            formatted_message = source_code.format(*args)
            renderable: ConsoleRenderable = Syntax(formatted_message, language)

            if box:
                renderable = Panel.fit(renderable, title=box_title)

            console.print(renderable)

    @staticmethod
    def verbose(message: str, *args, box: bool = False, box_title: Optional[str] = None):
        """
        Writes a log message at level `Log.Level.VERBOSE`.

        :param message:     The log message to be written
        :param args:        Optional arguments to be included in the log message
        :param box:         True, if a box should be surrounded by a box, False otherwise
        :param box_title:   An optional title to be printed at the top of the box surrounding the log message
        """
        log_level = logging.DEBUG

        if logging.getLogger().isEnabledFor(log_level):
            formatted_message = LogHandler.format_message(message.format(*args), log_level)
            style = LogHandler.get_style(log_level)

            if box:
                text = Text(formatted_message, style=style) if style else Text(formatted_message)
                console.print(Panel.fit(text, title=box_title))
            else:
                console.print(formatted_message, style=style)
