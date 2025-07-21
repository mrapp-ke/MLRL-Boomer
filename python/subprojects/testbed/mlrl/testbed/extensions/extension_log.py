"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow configuring the logger.
"""
import logging as log
import sys

from argparse import Namespace
from enum import Enum
from typing import Set, override

from mlrl.testbed.experiments.experiment import Experiment
from mlrl.testbed.extensions.extension import Extension

from mlrl.util.cli import Argument, EnumArgument


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

    LOG_LEVEL = EnumArgument(
        '--log-level',
        enum=LogLevel,
        default=LogLevel.INFO,
        description='The log level to be used.',
    )

    @override
    def _get_arguments(self) -> Set[Argument]:
        """
        See :func:`mlrl.testbed.extensions.extension.Extension._get_arguments`
        """
        return {self.LOG_LEVEL}

    @override
    def configure_experiment(self, args: Namespace, _: Experiment.Builder):
        """
        See :func:`mlrl.testbed.extensions.extension.Extension.configure_experiment`
        """
        log_level = self.LOG_LEVEL.get_value(args).value
        root = log.getLogger()
        root.setLevel(log_level)
        out_handler = log.StreamHandler(sys.stdout)
        out_handler.setLevel(log_level)
        out_handler.setFormatter(log.Formatter('%(message)s'))
        existing_handlers = list(root.handlers)

        for existing_handler in existing_handlers:
            root.removeHandler(existing_handler)

        root.addHandler(out_handler)
