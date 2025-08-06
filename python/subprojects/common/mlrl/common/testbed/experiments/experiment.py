"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for performing experiments using rule learning algorithms.
"""
from typing import Any, Dict, Optional

from sklearn.base import BaseEstimator

from mlrl.common.mixins import NominalFeatureSupportMixin, OrdinalFeatureSupportMixin

from mlrl.testbed_sklearn.experiments import SkLearnExperiment
from mlrl.testbed_sklearn.experiments.dataset import AttributeType, TabularDataset

from mlrl.testbed.experiments.experiment import Experiment
from mlrl.testbed.experiments.input.dataset.splitters.splitter import DatasetSplitter
from mlrl.testbed.experiments.state import ExperimentState
from mlrl.testbed.experiments.timer import Timer


class RuleLearnerExperiment(SkLearnExperiment):
    """
    An experiment that trains and evaluates a rule model using the scikit-learn framework.
    """

    class Builder(SkLearnExperiment.Builder):
        """
        Allows to configure and create instances of the class `RuleLearnerExperiment`.
        """

        def _create_experiment(self, initial_state: ExperimentState, dataset_splitter: DatasetSplitter) -> Experiment:
            return RuleLearnerExperiment(initial_state=initial_state, dataset_splitter=dataset_splitter)

    def _fit(self, estimator: BaseEstimator, dataset: TabularDataset,
             fit_kwargs: Optional[Dict[str, Any]]) -> Timer.Duration:
        fit_kwargs = fit_kwargs if fit_kwargs else {}

        # Set the indices of ordinal features, if supported...
        if isinstance(estimator, OrdinalFeatureSupportMixin):
            fit_kwargs[OrdinalFeatureSupportMixin.KWARG_ORDINAL_FEATURE_INDICES] = dataset.get_feature_indices(
                AttributeType.ORDINAL)

        # Set the indices of nominal features, if supported...
        if isinstance(estimator, NominalFeatureSupportMixin):
            fit_kwargs[NominalFeatureSupportMixin.KWARG_NOMINAL_FEATURE_INDICES] = dataset.get_feature_indices(
                AttributeType.NOMINAL)

        return super()._fit(estimator, dataset, fit_kwargs)
