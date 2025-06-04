"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow configuring the functionality to preprocess datasets.
"""
from argparse import Namespace
from typing import List

from mlrl.testbed.experiments.input.dataset.preprocessors.preprocessor import Preprocessor
from mlrl.testbed.experiments.input.dataset.preprocessors.tabular.one_hot_encoder import OneHotEncoder
from mlrl.testbed.extensions.extension import Extension

from mlrl.util.cli import Argument, BoolArgument


class PreprocessorExtension(Extension):
    """
    An extension that configures the functionality to preprocess datasets.
    """

    ONE_HOT_ENCODING = BoolArgument(
        '--one-hot-encoding',
        default=False,
        description='Whether one-hot-encoding should be used to encode nominal features or not.',
    )

    def _get_arguments(self) -> List[Argument]:
        """
        See :func:`mlrl.testbed.extensions.extension.Extension._get_arguments`
        """
        return [self.ONE_HOT_ENCODING]

    @staticmethod
    def get_preprocessors(args: Namespace) -> List[Preprocessor]:
        """
        Returns the preprocessors to be used for preprocessing datasets according to the configuration.

        :param args:    The command line arguments specified by the user
        :return:        The preprocessors to be used
        """
        preprocessors = []

        if PreprocessorExtension.ONE_HOT_ENCODING.get_value(args):
            preprocessors.append(OneHotEncoder())

        return preprocessors
