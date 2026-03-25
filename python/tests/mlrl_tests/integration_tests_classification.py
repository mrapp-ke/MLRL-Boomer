"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)
"""

from abc import ABC

import pytest

from .datasets import Dataset
from .integration_tests import IntegrationTests


class ClassificationIntegrationTests(IntegrationTests, ABC):
    """
    Defines a series of integration tests for any type of rule learning algorithm that can be applied to classification
    problems.
    """

    @pytest.fixture
    def dataset(self) -> Dataset:
        return Dataset()
