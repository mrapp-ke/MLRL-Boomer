"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for writing log messages.
"""

import contextlib
import logging
import os
from contextvars import ContextVar

from contextlib import contextmanager
from typing import override

from rich.console import Console, ConsoleRenderable, ConsoleOptions
from rich.logging import LogRecord, RichHandler
from rich.panel import Panel
from rich.style import Style
from rich.segment import Segment
from rich.syntax import Syntax
from rich.text import Text

from mlrl.testbed.util.io import ENCODING_UTF8

WIDTH: int | None = None

PLAIN: bool = False


class IndentationLevel:
    """
    Used to manage the indentation level of log messages.
    """

    class IndentedRenderable:
        """
        Wraps a `ConsoleRenderable` and adds indentation, depending on the current level.
        """

        def __init__(self, renderable: ConsoleRenderable, level: int):
            """
            :param renderable:  The `ConsoleRenderable` to be wrapped
            :param level:       The indentation level
            """
            self._renderable = renderable
            self._level = level

        def __rich_console__(self, console: Console, options: ConsoleOptions):
            level = self._level
            prefix = IndentationLevel.get_prefix(level)
            max_width = max(1, options.max_width - len(prefix))
            at_line_start = True

            for segment in console.render(self._renderable, options.update_width(max_width)):
                if segment.is_control:
                    yield segment
                else:
                    text = segment.text
                    style = segment.style

                    while '\n' in text:
                        current_index = text.index('\n')
                        current_text = text[:current_index]

                        if at_line_start:
                            yield Segment(prefix, style=IndentationLevel.PREFIX_STYLE)

                        if current_text:
                            yield Segment(current_text, style)

                        yield Segment('\n')
                        at_line_start = True
                        text = text[current_index + 1 :]

                    if text:
                        if at_line_start:
                            yield Segment(prefix, style=IndentationLevel.PREFIX_STYLE)
                            at_line_start = False

                        yield Segment(text, style)

            if not at_line_start:
                yield Segment('\n')

    PREFIX_STYLE = Style(color='grey50')

    def __init__(self, level: int):
        """
        :param level:   The indentation level
        """
        self.level = level

    def increase(self):
        """
        Increase the indentation level.
        """
        self.level += 1

    def decrease(self):
        """
        Decreases the indentation level.
        """
        if self.level > 0:
            self.level -= 1

    def print(self, message: str, style: Style | None = None, box: bool = False, box_title: str | None = None):
        renderable: ConsoleRenderable = Text(message, style=style) if style else Text(message)
        console = get_console()
        level = self.level

        if box:
            if PLAIN:
                if box_title:
                    title = f'{box_title}:\n'
                    title_renderable = Text(title, style=style) if style else Text(title)
                    console.print(IndentationLevel.IndentedRenderable(title_renderable, level=level))
            else:
                renderable = self.decorate_with_box(renderable, box_title=box_title)

        console.print(IndentationLevel.IndentedRenderable(renderable, level=level))

    @staticmethod
    def decorate_with_box(renderable: ConsoleRenderable, box_title: str | None = None) -> ConsoleRenderable:
        return Panel.fit(renderable, title=Text(box_title, style=Style(bold=True)) if box_title else None)

    @staticmethod
    def get_prefix(level: int, prefix: str = '│') -> str:
        """
        Returns the prefix to be used for log messages at a specific indentation level.

        :param level:   The indentation level
        :param prefix:  A text that should be printed before the log message
        :return:        A prefix
        """
        text = ''

        for i in range(level):
            symbol = prefix if i == level - 1 else '│'
            text += f' {symbol}'

        return f'{text} '


INDENTATION_LEVEL: ContextVar[IndentationLevel] = ContextVar('indentation_level', default=IndentationLevel(level=0))


def get_console() -> Console:
    """
    Returns the console to be used for logging.
    """
    return Console(width=WIDTH, color_system=None if PLAIN else 'auto')


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
        logging.DEBUG: Style(color='grey50'),
        logging.WARNING: Style(color='yellow'),
        logging.ERROR: Style(color='red', bold=True),
    }

    SYMBOL_PER_LOG_LEVEL = {
        logging.DEBUG: '○',
        logging.WARNING: '⚠',
        logging.ERROR: '✗',
    }

    def __init__(self):
        super().__init__(console=get_console(), show_time=False, show_level=False, show_path=False)

    @staticmethod
    def format_message(message: str, log_level: int) -> str:
        """
        Formats a given log message, depending on a given log level.

        :param message:     The log message to be formatted
        :param log_level:   The log level
        :return:            The formatted message
        """
        symbol = LogHandler.SYMBOL_PER_LOG_LEVEL.get(log_level, '')
        return symbol + (' ' if symbol else '') + message

    @staticmethod
    def get_style(log_level: int) -> Style | None:
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
        message = self.format_message(message, log_level)
        message_style = self.get_style(log_level)
        formatted_message = Text(message, style=message_style) if message_style else Text(message)
        indentation_level = INDENTATION_LEVEL.get()
        level = indentation_level.level
        prefix = Text(IndentationLevel.get_prefix(level), style=IndentationLevel.PREFIX_STYLE)
        return prefix + formatted_message


class Log:
    """
    Allows to write log messages.
    """

    @staticmethod
    @contextmanager
    def indented():
        """
        A context manager that indents all log messages emitted within the block. Nesting multiple context managers
        results in deeper indentation levels.
        """
        indentation_level = INDENTATION_LEVEL.get()
        console = get_console()

        try:
            indentation_level.increase()
            prefix = IndentationLevel.get_prefix(level=indentation_level.level, prefix='╭')
            console.print(Text(prefix, style=IndentationLevel.PREFIX_STYLE))
            yield
        finally:
            prefix = IndentationLevel.get_prefix(level=indentation_level.level, prefix='╰')
            console.print(Text(prefix, style=IndentationLevel.PREFIX_STYLE))
            indentation_level.decrease()
            prefix = IndentationLevel.get_prefix(level=indentation_level.level)
            console.print(Text(prefix), style=IndentationLevel.PREFIX_STYLE)

    @staticmethod
    def error(message: str, *args, error: Exception | None = None, box: bool = False, box_title: str | None = None):
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
            INDENTATION_LEVEL.get().print(message=formatted_message, style=style, box=box, box_title=box_title)

            if error:
                get_console().print_exception(extra_lines=2)

    @staticmethod
    def warning(message: str, box: bool = False, box_title: str | None = None):
        """
        Writes a log message at level `Log.Level.WARNING`.

        :param message:     The log message to be written
        :param box:         True, if a box should be surrounded by a box, False otherwise
        :param box_title:   An optional title to be printed at the top of the box surrounding the log message
        """
        log_level = logging.WARNING

        if logging.getLogger().isEnabledFor(log_level):
            formatted_message = LogHandler.format_message(message, log_level)
            style = LogHandler.get_style(log_level)
            INDENTATION_LEVEL.get().print(message=formatted_message, style=style, box=box, box_title=box_title)

    @staticmethod
    def success(message: str, box: bool = False, box_title: str | None = None):
        """
        Writes a log message at level `Log.Level.INFO` indicating successful operation of an operation.

        :param message:     The log message to be written
        :param args:        Optional arguments to be included in the log message
        :param box:         True, if a box should be surrounded by a box, False otherwise
        :param box_title:   An optional title to be printed at the top of the box surrounding the log message
        """
        log_level = logging.INFO

        if logging.getLogger().isEnabledFor(log_level):
            formatted_message = f'✓ {message}'
            style = Style(color='green', bold=True)
            INDENTATION_LEVEL.get().print(message=formatted_message, style=style, box=box, box_title=box_title)

    @staticmethod
    def info(message: str, box: bool = False, box_title: str | None = None):
        """
        Writes a log message at level `Log.Level.INFO`.

        :param message:     The log message to be written
        :param box:         True, if a box should be surrounded by a box, False otherwise
        :param box_title:   An optional title to be printed at the top of the box surrounding the log message
        """
        log_level = logging.INFO

        if logging.getLogger().isEnabledFor(log_level):
            formatted_message = LogHandler.format_message(message, log_level)
            style = LogHandler.get_style(log_level)
            INDENTATION_LEVEL.get().print(message=formatted_message, style=style, box=box, box_title=box_title)

    @staticmethod
    def separator(title: str):
        """
        Writes a log messages that acts as a separator with a specific title.

        :param title: The title to be used
        """
        log_level = logging.INFO

        if logging.getLogger().isEnabledFor(log_level):
            if PLAIN:
                indentation_level = INDENTATION_LEVEL.get()
                indentation_level.print('')
                indentation_level.print(f'{title}:')
                indentation_level.print('')
            else:
                console = get_console()
                console.print('')
                console.rule(title)
                console.print('')

    @staticmethod
    def source_code(source_code: str, language: str, box_title: str | None = None):
        """
        Writes a log message containing source code in a specific language at level `Log.Level.INFO`.

        :param source_code: The source code to be written
        :param args:        Optional arguments to be included in the log message
        :param language:    The language used by the source code
        :param box_title:   An optional title to be printed at the top of the box surrounding the log message
        """
        log_level = logging.INFO

        if logging.getLogger().isEnabledFor(log_level):
            renderable: ConsoleRenderable = Syntax(source_code, language, word_wrap=True)
            renderable = IndentationLevel.decorate_with_box(renderable, box_title=box_title)
            indentation_level = INDENTATION_LEVEL.get()
            get_console().print(IndentationLevel.IndentedRenderable(renderable, level=indentation_level.level))

    @staticmethod
    def verbose(message: str, box: bool = False, box_title: str | None = None):
        """
        Writes a log message at level `Log.Level.VERBOSE`.

        :param message:     The log message to be written
        :param box:         True, if a box should be surrounded by a box, False otherwise
        :param box_title:   An optional title to be printed at the top of the box surrounding the log message
        """
        log_level = logging.DEBUG

        if logging.getLogger().isEnabledFor(log_level):
            formatted_message = LogHandler.format_message(message, log_level)
            style = LogHandler.get_style(log_level)
            INDENTATION_LEVEL.get().print(message=formatted_message, style=style, box=box, box_title=box_title)
