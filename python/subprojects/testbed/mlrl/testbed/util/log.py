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
            logging.error((message + ': {}').format(*args, error))
        else:
            logging.error(message.format(*args))

    @staticmethod
    def warning(message: str, *args):
        """
        Writes a log message at level `Log.Level.WARNING`.

        :param message: The log message to be written
        :param args:    Optional arguments to be included in the log message
        """
        logging.warning(message.format(*args))

    @staticmethod
    def success(message: str, *args):
        """
        Writes a log message at level `Log.Level.INFO` indicating successful operation of an operation.

        :param message: The log message to be written
        :param args:    Optional arguments to be included in the log message
        """
        Log.info('✓ ' + message.format(*args))

    @staticmethod
    def info(message: str, *args):
        """
        Writes a log message at level `Log.Level.INFO`.

        :param message: The log message to be written
        :param args:    Optional arguments to be included in the log message
        """
        logging.info(message.format(*args))

    @staticmethod
    def verbose(message: str, *args):
        """
        Writes a log message at level `Log.Level.VERBOSE`.

        :param message: The log message to be written
        :param args:    Optional arguments to be included in the log message
        """
        logging.debug(message.format(*args))
