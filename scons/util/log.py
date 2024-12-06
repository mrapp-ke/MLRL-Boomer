"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for writing log messages.
"""
import logging
import sys

from enum import Enum
from typing import Optional


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
    def error(message: str, *args, error: Optional[Exception] = None, exit_code: int = 1):
        """
        Writes a log message at level `Log.Level.ERROR` and terminates the build system.

        :param message:     The log message to be written
        :param args:        Optional arguments to be included in the log message
        :param error:       An optional error to be included in the log message
        :param exit_code:   The exit code to be returned when terminating the build system
        """
        if error:
            logging.error(message + ': %s', *args, error)
        else:
            logging.error(message, *args)

        sys.exit(exit_code)

    @staticmethod
    def warning(message: str, *args):
        """
        Writes a log message at level `Log.Level.WARNING`.

        :param message: The log message to be written
        :param args:    Optional arguments to be included in the log message
        """
        logging.warning(message, *args)

    @staticmethod
    def info(message: str, *args):
        """
        Writes a log message at level `Log.Level.INFO`.

        :param message: The log message to be written
        :param args:    Optional arguments to be included in the log message
        """
        logging.info(message, *args)

    @staticmethod
    def verbose(message: str, *args):
        """
        Writes a log message at level `Log.Level.VERBOSE`.

        :param message: The log message to be written
        :param args:    Optional arguments to be included in the log message
        """
        logging.debug(message, *args)
