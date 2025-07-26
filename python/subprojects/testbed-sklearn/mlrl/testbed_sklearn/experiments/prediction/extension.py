"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow configuring the functionality to obtain predictions from a machine learning model.
"""
from argparse import Namespace
from typing import Set, override

from mlrl.testbed.experiments.prediction_type import PredictionType
from mlrl.testbed.extensions.extension import Extension

from mlrl.util.cli import Argument, EnumArgument


class PredictionTypeExtension(Extension):
    """
    An extension that configures the type of predictions to be obtained from a machine learning model.
    """

    PREDICTION_TYPE = EnumArgument(
        '--prediction-type',
        enum=PredictionType,
        default=PredictionType.BINARY,
        description='The type of predictions that should be obtained from the learner.',
    )

    @override
    def _get_arguments(self) -> Set[Argument]:
        """
        See :func:`mlrl.testbed.extensions.extension.Extension._get_arguments`
        """
        return {self.PREDICTION_TYPE}

    @staticmethod
    def get_prediction_type(args: Namespace) -> PredictionType:
        """
        Returns the `PredictionType` to be used for obtaining predictions from a machine learning model according to the
        configuration.

        :param args:    The command line arguments specified by the user
        :return:        The `PredictionType` to be used
        """
        return PredictionTypeExtension.PREDICTION_TYPE.get_value(args)
