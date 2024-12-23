"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from abc import ABC
from unittest import SkipTest

from .cmd_builder import DATASET_ATP7D, DATASET_ATP7D_BINARY, DATASET_ATP7D_MEKA, DATASET_ATP7D_NOMINAL, \
    DATASET_ATP7D_NUMERICAL_SPARSE, DATASET_ATP7D_ORDINAL, DATASET_HOUSING
from .integration_tests import IntegrationTests


class RegressionIntegrationTests(IntegrationTests, ABC):
    """
    Defines a series of integration tests for any type of rule learning algorithm that can be applied to regression
    problems.
    """

    # pylint: disable=invalid-name
    def __init__(self,
                 dataset_default: str = DATASET_ATP7D,
                 dataset_numerical_sparse: str = DATASET_ATP7D_NUMERICAL_SPARSE,
                 dataset_binary: str = DATASET_ATP7D_BINARY,
                 dataset_nominal: str = DATASET_ATP7D_NOMINAL,
                 dataset_ordinal: str = DATASET_ATP7D_ORDINAL,
                 dataset_single_output: str = DATASET_HOUSING,
                 dataset_meka: str = DATASET_ATP7D_MEKA,
                 methodName='runTest'):
        super().__init__(dataset_default=dataset_default,
                         dataset_numerical_sparse=dataset_numerical_sparse,
                         dataset_binary=dataset_binary,
                         dataset_nominal=dataset_nominal,
                         dataset_ordinal=dataset_ordinal,
                         dataset_single_output=dataset_single_output,
                         dataset_meka=dataset_meka,
                         methodName=methodName)

    @classmethod
    def setUpClass(cls):
        """
        Sets up the test class.
        """
        if cls is RegressionIntegrationTests:
            raise SkipTest(cls.__name__ + ' is an abstract base class')

        super().setUpClass()

    def test_single_output_regression(self):
        """
        Tests the evaluation of the rule learning algorithm when predicting regression scores for a single-output
        problem.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_single_output) \
            .print_evaluation()
        builder.run_cmd('single-output-regression')
