"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)
"""

import pytest

from ..cmd_runner import CmdRunner
from ..integration_tests import IntegrationTests

from mlrl.testbed.experiments.input.dataset.splitters.arguments import DatasetSplitterArguments

from mlrl.util.options import Options


class MlrlTestbedSklearnIntegrationTestsMixin(IntegrationTests):
    """
    A mixin for integration tests for the mlrl-testbed-sklearn package.
    """

    @pytest.mark.parametrize(
        'data_split, data_split_options',
        [
            (DatasetSplitterArguments.VALUE_TRAIN_TEST, Options()),
            (DatasetSplitterArguments.VALUE_CROSS_VALIDATION, Options()),
            (
                DatasetSplitterArguments.VALUE_CROSS_VALIDATION,
                Options({DatasetSplitterArguments.OPTION_FIRST_FOLD: 1, DatasetSplitterArguments.OPTION_LAST_FOLD: 1}),
            ),
        ],
    )
    def test_data_splitter(self, data_split: str, data_split_options: Options):
        test_name = f'data-splitter_{data_split}' + (f'_{data_split_options}' if data_split_options else '')
        builder = self.create_cmd_builder().data_split(data_split, options=data_split_options)
        CmdRunner(builder).run(test_name)
