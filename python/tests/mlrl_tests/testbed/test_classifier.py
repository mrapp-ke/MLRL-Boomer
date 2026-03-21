"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from typing import Any, override

import pytest

from ..datasets import Dataset
from .cmd_builder_classification import ClassificationTestbedCmdBuilder
from .integration_tests import TestbedIntegrationTestsMixin
from .integration_tests_classification import SklearnTestbedClassificationIntegrationTests


@pytest.mark.testbed
@pytest.mark.classification
class TestTestbedClassifier(SklearnTestbedClassificationIntegrationTests, TestbedIntegrationTestsMixin):
    """
    Defines a series of integration tests for the package mlrl-testbed when applied to classification problems.
    """

    @override
    def create_cmd_builder(self, dataset: str = Dataset.EMOTIONS) -> Any:
        return ClassificationTestbedCmdBuilder(dataset=dataset)
