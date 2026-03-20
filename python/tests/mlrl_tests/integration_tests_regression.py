"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
# pylint: disable=missing-function-docstring
from abc import ABC

import pytest

from .cmd_runner import CmdRunner
from .datasets import Dataset
from .integration_tests import IntegrationTests


class RegressionIntegrationTests(IntegrationTests, ABC):
    """
    Defines a series of integration tests for any type of rule learning algorithm that can be applied to regression
    problems.
    """

    @pytest.fixture
    def dataset(self) -> Dataset:
        return Dataset(default=Dataset.ATP7D,
                       numerical=Dataset.ATP7D,
                       numerical_sparse=Dataset.ATP7D_NUMERICAL_SPARSE,
                       binary=Dataset.ATP7D_BINARY,
                       nominal=Dataset.ATP7D_NOMINAL,
                       ordinal=Dataset.ATP7D_ORDINAL,
                       single_output=Dataset.HOUSING,
                       meka=Dataset.ATP7D_MEKA,
                       svm=Dataset.BODYFAT)

    def test_single_output_regression(self, dataset: Dataset):
        builder = self._create_cmd_builder(dataset=dataset.single_output) \
            .print_evaluation()
        CmdRunner(builder).run('single-output-regression')
