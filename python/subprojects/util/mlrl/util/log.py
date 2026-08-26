"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for writing log messages.
"""

import logging


class Log:
    """
    Allows to write log messages.
    """

    @staticmethod
    def error(
        message: str,
    ):
        """
        Writes a log message at level `Log.Level.ERROR` and terminates the build system.

        :param message: The log message to be written
        """
        logging.getLogger(__name__).error(message)

    @staticmethod
    def warning(message: str):
        """
        Writes a log message at level `Log.Level.WARNING`.

        :param message: The log message to be written
        """
        logging.getLogger(__name__).warning(message)

    @staticmethod
    def info(message: str):
        """
        Writes a log message at level `Log.Level.INFO`.

        :param message: The log message to be written
        """
        logging.getLogger(__name__).info(message)

    @staticmethod
    def verbose(message: str):
        """
        Writes a log message at level `Log.Level.VERBOSE`.

        :param message: The log message to be written
        """
        logging.getLogger(__name__).debug(message)
