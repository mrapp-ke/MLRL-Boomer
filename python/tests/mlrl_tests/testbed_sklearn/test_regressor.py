"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
# pylint: disable=missing-function-docstring
from typing import Any, override

import pytest

from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import RegressorChain

from ..common.cmd_runner import CmdRunner
from ..common.datasets import Dataset
from ..common.integration_tests_regression import RegressionIntegrationTests
from .cmd_builder_regression import SkLearnRegressorCmdBuilder


@pytest.mark.sklearn
@pytest.mark.regression
class TestSkLearnRegressor(RegressionIntegrationTests):
    """
    Defines a series of integration tests for a scikit-learn classifier.
    """

    @override
    def _create_cmd_builder(self, dataset: str = Dataset.ATP7D) -> Any:
        return SkLearnRegressorCmdBuilder(RandomForestRegressor,
                                          dataset=dataset).add_algorithmic_argument('--n-estimators', 1)

    def test_meta_estimator(self):
        builder = self._create_cmd_builder() \
            .meta_estimator(RegressorChain) \
            .add_algorithmic_argument('--meta-verbose', 'True') \
            .print_evaluation() \
            .save_evaluation(False)
        CmdRunner(builder).run('meta-estimator')
