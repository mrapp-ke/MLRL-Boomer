"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for writing log messages.
"""

from typing import Optional

from rich.console import Console

console = Console(soft_wrap=True)


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
        if error:
            message = message + ': {}'.format(*args, error)
        else:
            message = message.format(*args)

        console.print(message)

    @staticmethod
    def warning(message: str, *args):
        """
        Writes a log message at level `Log.Level.WARNING`.

        :param message: The log message to be written
        :param args:    Optional arguments to be included in the log message
        """
        console.print('WARNING: ' + message.format(*args))

    @staticmethod
    def success(message: str, *args):
        """
        Writes a log message at level `Log.Level.INFO` indicating successful operation of an operation.

        :param message: The log message to be written
        :param args:    Optional arguments to be included in the log message
        """
        console.print('✓ ' + message.format(*args))

    @staticmethod
    def info(message: str, *args):
        """
        Writes a log message at level `Log.Level.INFO`.

        :param message: The log message to be written
        :param args:    Optional arguments to be included in the log message
        """
        console.print(message.format(*args))

    @staticmethod
    def verbose(message: str, *args):
        """
        Writes a log message at level `Log.Level.VERBOSE`.

        :param message: The log message to be written
        :param args:    Optional arguments to be included in the log message
        """
        console.print('DEBUG: ' + message.format(*args))
