"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for writing log messages.
"""
import logging

from typing import Optional


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
            logging.error(message + ': %s', *args, error)
        else:
            logging.error(message, *args)

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
