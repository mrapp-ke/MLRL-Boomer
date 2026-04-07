"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Defines command line arguments for configuring the functionality to write log messages.
"""

import logging

from argparse import Namespace
from enum import Enum

from mlrl.testbed.log import log
from mlrl.testbed.log.log import LogHandler

from mlrl.util.cli import EnumArgument, FlagArgument, IntArgument


class LogLevel(Enum):
    """
    Specifies all valid textual representations of log levels.
    """

    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR


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
        '--log-width',
        description='The width of the console output.',
    )

    LOG_PLAIN = FlagArgument(
        '--log-plain',
        description='Restrict the console output to plain text without any colors or structural elements.',
    )


def configure_logger(args: Namespace):
    """
    Configures the logger according to the command line arguments provided by the user.

    :param args: The command line arguments provided by the user
    """
    log.WIDTH = LogArguments.LOG_WIDTH.get_value(args)
    log.PLAIN = LogArguments.LOG_PLAIN.get_value(args, default=False)
    logger = logging.getLogger()

    for existing_handler in logger.handlers:
        logger.removeHandler(existing_handler)

    logger.addHandler(LogHandler())
    logger.setLevel(LogArguments.LOG_LEVEL.get_value(args).value)
