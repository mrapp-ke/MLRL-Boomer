"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for writing log messages.
"""

import logging
import sys

from enum import Enum


class Log:
    """
    Allows to write log messages.
    """

    class Level(Enum):
        """
        The log levels supported by the build system.
        """

        NONE = logging.NOTSET
        ERROR = logging.ERROR
        WARNING = logging.WARNING
        INFO = logging.INFO
        VERBOSE = logging.DEBUG

    @staticmethod
    def configure(log_level: Level = Level.INFO):
        """
        Configures the logger to be used by the build system.

        :param log_level: The log level to be used
        """
        root = logging.getLogger()
        root.setLevel(log_level.value)
        out_handler = logging.StreamHandler(sys.stdout)
        out_handler.setLevel(log_level.value)
        out_handler.setFormatter(logging.Formatter('%(message)s'))
        root.addHandler(out_handler)

    @staticmethod
    def error(message: str, error: Exception | None = None, exit_code: int = 1):
        """
        Writes a log message at level `Log.Level.ERROR` and terminates the build system.

        :param message:     The log message to be written
        :param args:        Optional arguments to be included in the log message
        :param error:       An optional error to be included in the log message
        :param exit_code:   The exit code to be returned when terminating the build system
        """
        logging.error(f'{message}: {error}' if error else message)
        sys.exit(exit_code)

    @staticmethod
    def warning(message: str):
        """
        Writes a log message at level `Log.Level.WARNING`.

        :param message: The log message to be written
        """
        logging.warning(message)

    @staticmethod
    def info(message: str):
        """
        Writes a log message at level `Log.Level.INFO`.

        :param message: The log message to be written
        """
        logging.info(message)

    @staticmethod
    def verbose(message: str):
        """
        Writes a log message at level `Log.Level.VERBOSE`.

        :param message: The log message to be written
        """
        logging.debug(message)
