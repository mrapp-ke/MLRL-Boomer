"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Defines command line arguments for configuring the functionality to write log messages.
"""
import logging

from argparse import Namespace
from enum import Enum

from mlrl.testbed.log import log
from mlrl.testbed.log.log import LogHandler

from mlrl.util.cli import EnumArgument, IntArgument


class LogLevel(Enum):
    """
    Specifies all valid textual representations of log levels.
    """
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARN = logging.WARN
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL
    FATAL = logging.FATAL
    NOTSET = logging.NOTSET


class LogArguments:
    """
    Defines command line arguments for configuring the functionality to write log messages.
    """

    LOG_LEVEL = EnumArgument(
        '--log-level',
        enum=LogLevel,
        default=LogLevel.INFO,
        description='The log level to be used.',
    )

    LOG_WIDTH = IntArgument(
        "--log-width",
        description='The width of the console output.',
    )


def configure_logger(args: Namespace):
    """
    Configures the logger according to the command line arguments provided by the user.

    :param args: The command line arguments provided by the user
    """
    log.WIDTH = LogArguments.LOG_WIDTH.get_value(args)
    logger = logging.getLogger()

    for existing_handler in logger.handlers:
        logger.removeHandler(existing_handler)

    logger.addHandler(LogHandler())
    logger.setLevel(LogArguments.LOG_LEVEL.get_value(args).value)
