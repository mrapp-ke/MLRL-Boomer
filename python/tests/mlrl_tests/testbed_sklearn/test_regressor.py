"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)
"""

from typing import Any, override

import pytest

from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import RegressorChain

from ..cmd_runner import CmdRunner
from ..datasets import Dataset
from ..integration_tests_regression import RegressionIntegrationTests
from ..testbed.integration_tests import MlrlTestbedIntegrationTestsMixin
from .cmd_builder_regression import SkLearnRegressorCmdBuilder
from .integration_tests import MlrlTestbedSklearnIntegrationTestsMixin


@pytest.mark.sklearn
@pytest.mark.regression
class TestSkLearnRegressor(
    RegressionIntegrationTests, MlrlTestbedIntegrationTestsMixin, MlrlTestbedSklearnIntegrationTestsMixin
):
    """
    Defines a series of integration tests for a scikit-learn classifier.
    """

    @override
    def create_cmd_builder(self, dataset: str = Dataset.ATP7D) -> Any:
        return SkLearnRegressorCmdBuilder(RandomForestRegressor, dataset=dataset).add_algorithmic_argument(
            '--n-estimators', 1
        )

    def test_meta_estimator(self):
        builder = (
            self.create_cmd_builder()
            .meta_estimator(RegressorChain)
            .add_algorithmic_argument('--meta-verbose', 'False')
            .print_evaluation()
            .save_evaluation(False)
        )
        CmdRunner(builder).run('meta-estimator')
