"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
# pylint: disable=missing-function-docstring
from typing import Any, override

import pytest

from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import ClassifierChain

from ..cmd_runner import CmdRunner
from ..datasets import Dataset
from ..integration_tests_classification import ClassificationIntegrationTests
from .cmd_builder_classification import SkLearnClassifierCmdBuilder
from .integration_tests import SklearnTestbedIntegrationTestsMixin


@pytest.mark.sklearn
@pytest.mark.classification
class TestSkLearnClassifier(ClassificationIntegrationTests, SklearnTestbedIntegrationTestsMixin):
    """
    Defines a series of integration tests for a scikit-learn classifier.
    """

    @override
    def _create_cmd_builder(self, dataset: str = Dataset.EMOTIONS) -> Any:
        return SkLearnClassifierCmdBuilder(RandomForestClassifier,
                                           dataset=dataset).add_algorithmic_argument('--n-estimators', 1)

    def test_meta_estimator(self):
        builder = self._create_cmd_builder() \
            .meta_estimator(ClassifierChain) \
            .add_algorithmic_argument('--meta-verbose', 'True') \
            .print_evaluation() \
            .save_evaluation(False)
        CmdRunner(builder).run('meta-estimator')
