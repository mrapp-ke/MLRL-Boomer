"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)
"""

from typing import Any, override

import pytest

from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import ClassifierChain

from ..cmd_runner import CmdRunner
from ..datasets import Dataset
from ..testbed.integration_tests import MlrlTestbedIntegrationTestsMixin
from .cmd_builder_classification import SkLearnClassifierCmdBuilder
from .integration_tests import MlrlTestbedSklearnIntegrationTestsMixin
from .integration_tests_classification import MlrlTestbedSklearnClassificationIntegrationTests


@pytest.mark.sklearn
@pytest.mark.classification
class TestSkLearnClassifier(
    MlrlTestbedSklearnClassificationIntegrationTests,
    MlrlTestbedIntegrationTestsMixin,
    MlrlTestbedSklearnIntegrationTestsMixin,
):
    """
    Defines a series of integration tests for a scikit-learn classifier.
    """

    @override
    def create_cmd_builder(self, dataset: str = Dataset.EMOTIONS) -> Any:
        return SkLearnClassifierCmdBuilder(RandomForestClassifier, dataset=dataset).add_algorithmic_argument(
            '--n-estimators', 1
        )

    def test_meta_estimator(self):
        builder = (
            self.create_cmd_builder()
            .meta_estimator(ClassifierChain)
            .add_algorithmic_argument('--meta-verbose', 'True')
            .print_evaluation()
            .save_evaluation(False)
        )
        CmdRunner(builder).run('meta-estimator')
