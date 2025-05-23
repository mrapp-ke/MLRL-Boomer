"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow configuring the logger.
"""
import logging as log

from argparse import ArgumentParser
from enum import Enum

from mlrl.testbed.profiles.profile import Profile

from mlrl.util.format import format_enum_values, format_set


class LogProfile(Profile):
    """
    A profile that configures the logger to be used by the command line API.
    """

    class LogLevel(Enum):
        """
        Specifies all valid textual representations of log levels.
        """
        DEBUG = ('debug', log.DEBUG)
        INFO = ('info', log.INFO)
        WARN = ('warn', log.WARN)
        WARNING = ('warning', log.WARNING)
        ERROR = ('error', log.ERROR)
        CRITICAL = ('critical', log.CRITICAL)
        FATAL = ('fatal', log.FATAL)
        NOTSET = ('notset', log.NOTSET)

        @classmethod
        def parse(cls, text: str):
            """
            Parses a given text that represents a log level. If the given text does not represent a valid log level, a
            `ValueError` is raised.

            :param text:    The text to be parsed
            :return:        A log level, depending on the given text
            """
            lower_text = text.lower()

            for enum in cls:
                level_name, level_int = enum.value

                if level_name == lower_text:
                    return level_int

            raise ValueError()

    def configure_arguments(self, argument_parser: ArgumentParser):
        """
        See :func:`mlrl.testbed.profiles.profile.Profile.configure_arguments`
        """
        argument_parser.add_argument('--log-level',
                                     type=LogProfile.LogLevel.parse,
                                     default=LogProfile.LogLevel.INFO.value,
                                     help='The log level to be used. Must be one of '
                                     + format_set(log_level.value[0] for log_level in LogProfile.LogLevel) + '.')
