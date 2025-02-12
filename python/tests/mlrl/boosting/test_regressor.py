"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from typing import Any

from ..common.cmd_builder import DATASET_ATP7D
from ..common.decorators import skip_test_on_ci
from ..common.integration_tests_regression import RegressionIntegrationTests
from .cmd_builder import HEAD_TYPE_COMPLETE, HEAD_TYPE_PARTIAL_DYNAMIC, HEAD_TYPE_PARTIAL_FIXED, HEAD_TYPE_SINGLE, \
    LOSS_SQUARED_ERROR_DECOMPOSABLE, LOSS_SQUARED_ERROR_NON_DECOMPOSABLE
from .cmd_builder_regression import BoomerRegressorCmdBuilder
from .integration_tests import BoomerIntegrationTestsMixin


class BoomerRegressorIntegrationTests(RegressionIntegrationTests, BoomerIntegrationTestsMixin):
    """
    Defines a series of integration tests for the BOOMER algorithm for regression problems.
    """

    # pylint: disable=invalid-name
    def __init__(self, methodName='runTest'):
        super().__init__(methodName=methodName)

    def _create_cmd_builder(self, dataset: str = DATASET_ATP7D) -> Any:
        return BoomerRegressorCmdBuilder(self, dataset=dataset)

    def test_decomposable_single_output_heads(self):
        """
        Tests the BOOMER algorithm when using a decomposable loss function for the induction of rules with single-output
        heads.
        """
        builder = self._create_cmd_builder() \
            .loss(LOSS_SQUARED_ERROR_DECOMPOSABLE) \
            .head_type(HEAD_TYPE_SINGLE) \
            .print_model_characteristics()
        builder.run_cmd('decomposable-single-output-heads')

    def test_decomposable_complete_heads(self):
        """
        Tests the BOOMER algorithm when using a decomposable loss function for the induction of rules with complete
        heads.
        """
        builder = self._create_cmd_builder() \
            .loss(LOSS_SQUARED_ERROR_DECOMPOSABLE) \
            .head_type(HEAD_TYPE_COMPLETE) \
            .print_model_characteristics()
        builder.run_cmd('decomposable-complete-heads')

    def test_decomposable_partial_fixed_heads(self):
        """
        Tests the BOOMER algorithm when using a decomposable loss function for the induction of rules that predict for a
        number of labels
        """
        builder = self._create_cmd_builder() \
            .loss(LOSS_SQUARED_ERROR_DECOMPOSABLE) \
            .head_type(HEAD_TYPE_PARTIAL_FIXED) \
            .print_model_characteristics()
        builder.run_cmd('decomposable-partial-fixed-heads')

    def test_decomposable_partial_dynamic_heads(self):
        """
        Tests the BOOMER algorithm when using a decomposable loss function for the induction of rules that predict for a
        dynamically determined subset of the available labels.
        """
        builder = self._create_cmd_builder() \
            .loss(LOSS_SQUARED_ERROR_DECOMPOSABLE) \
            .head_type(HEAD_TYPE_PARTIAL_DYNAMIC) \
            .print_model_characteristics()
        builder.run_cmd('decomposable-partial-dynamic-heads')

    def test_non_decomposable_single_label_heads(self):
        """
        Tests the BOOMER algorithm when using a non-decomposable loss function for the induction of rules with
        single-output heads.
        """
        builder = self._create_cmd_builder() \
            .loss(LOSS_SQUARED_ERROR_NON_DECOMPOSABLE) \
            .head_type(HEAD_TYPE_SINGLE) \
            .print_model_characteristics()
        builder.run_cmd('non-decomposable-single-output-heads')

    @skip_test_on_ci
    def test_non_decomposable_complete_heads(self):
        """
        Tests the BOOMER algorithm when using a non-decomposable loss function for the induction of rules with complete
        heads.
        """
        builder = self._create_cmd_builder() \
            .loss(LOSS_SQUARED_ERROR_NON_DECOMPOSABLE) \
            .head_type(HEAD_TYPE_COMPLETE) \
            .print_model_characteristics()
        builder.run_cmd('non-decomposable-complete-heads')

    @skip_test_on_ci
    def test_non_decomposable_partial_fixed_heads(self):
        """
        Tests the BOOMER algorithm when using a non-decomposable loss function for the induction of rules that predict
        for a number of labels
        """
        builder = self._create_cmd_builder() \
            .loss(LOSS_SQUARED_ERROR_NON_DECOMPOSABLE) \
            .head_type(HEAD_TYPE_PARTIAL_FIXED) \
            .print_model_characteristics()
        builder.run_cmd('non-decomposable-partial-fixed-heads')

    @skip_test_on_ci
    def test_non_decomposable_partial_dynamic_heads(self):
        """
        Tests the BOOMER algorithm when using a non-decomposable loss function for the induction of rules that predict
        for a dynamically determined subset of the available labels.
        """
        builder = self._create_cmd_builder() \
            .loss(LOSS_SQUARED_ERROR_NON_DECOMPOSABLE) \
            .head_type(HEAD_TYPE_PARTIAL_DYNAMIC) \
            .print_model_characteristics()
        builder.run_cmd('non-decomposable-partial-dynamic-heads')
