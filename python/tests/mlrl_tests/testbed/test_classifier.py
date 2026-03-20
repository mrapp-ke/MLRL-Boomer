"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from typing import Any, override

import pytest

from ..datasets import Dataset
from ..integration_tests_classification import ClassificationIntegrationTests
from .cmd_builder_classification import TestbedClassifierCmdBuilder
from .integration_tests import TestbedIntegrationTestsMixin


@pytest.mark.testbed
@pytest.mark.classification
class TestTestbedClassifier(ClassificationIntegrationTests, TestbedIntegrationTestsMixin):
    """
    Defines a series of integration tests for the package mlrl-testbed when applied to classification problems.
    """

    @override
    def _create_cmd_builder(self, dataset: str = Dataset.EMOTIONS) -> Any:
        return TestbedClassifierCmdBuilder(dataset=dataset)
