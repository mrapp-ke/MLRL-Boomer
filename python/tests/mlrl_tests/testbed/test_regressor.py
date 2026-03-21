"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from typing import Any, override

import pytest

from ..datasets import Dataset
from ..integration_tests_regression import RegressionIntegrationTests
from .cmd_builder_regression import RegressionTestbedCmdBuilder
from .integration_tests import TestbedIntegrationTestsMixin


@pytest.mark.testbed
@pytest.mark.regression
class TestTestbedRegressor(RegressionIntegrationTests, TestbedIntegrationTestsMixin):
    """
    Defines a series of integration tests for the BOOMER algorithm for regression problems.
    """

    @override
    def _create_cmd_builder(self, dataset: str = Dataset.ATP7D) -> Any:
        return RegressionTestbedCmdBuilder(dataset=dataset)
