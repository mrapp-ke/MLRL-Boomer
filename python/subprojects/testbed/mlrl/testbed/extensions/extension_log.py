"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow configuring the logger.
"""
import logging as log
import sys

from argparse import Namespace
from enum import Enum
from typing import List

from mlrl.testbed.experiments.experiment import Experiment
from mlrl.testbed.extensions.extension import Extension

from mlrl.util.cli import Argument, SetArgument


class LogExtension(Extension):
    """
    An extension that configures the logger to be used by the command line API.
    """

    class LogLevel(Enum):
        """
        Specifies all valid textual representations of log levels.
        """
        DEBUG = log.DEBUG
        INFO = log.INFO
        WARN = log.WARN
        WARNING = log.WARNING
        ERROR = log.ERROR
        CRITICAL = log.CRITICAL
        FATAL = log.FATAL
        NOTSET = log.NOTSET

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
                if enum.name.lower() == lower_text:
                    return enum.value

            raise ValueError()

    LOG_LEVEL = SetArgument(
        '--log-level',
        values=LogLevel,
        default=LogLevel.INFO,
        help='The log level to be used.',
    )

    def get_arguments(self) -> List[Argument]:
        """
        See :func:`mlrl.testbed.extensions.extension.Extension.get_arguments`
        """
        return [self.LOG_LEVEL]

    def configure_experiment(self, args: Namespace, _: Experiment.Builder):
        """
        See :func:`mlrl.testbed.extensions.extension.Extension.configure_experiment`
        """
        value = self.LOG_LEVEL.get_value(args)
        log_level = LogExtension.LogLevel.parse(value) if value else None
        root = log.getLogger()
        root.setLevel(log_level)
        out_handler = log.StreamHandler(sys.stdout)
        out_handler.setLevel(log_level)
        out_handler.setFormatter(log.Formatter('%(levelname)s %(message)s'))
        root.addHandler(out_handler)
