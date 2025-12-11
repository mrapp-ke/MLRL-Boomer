"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for performing experiments using rule learning algorithms.
"""
from argparse import Namespace
from typing import Any, Dict, Optional, override

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

        def _create_experiment(self, args: Namespace, initial_state: ExperimentState,
                               dataset_splitter: DatasetSplitter) -> Experiment:
            return RuleLearnerExperiment(args=args, initial_state=initial_state, dataset_splitter=dataset_splitter)

    class TrainingProcedure(SkLearnExperiment.TrainingProcedure):
        """
        Allows to fit a rule learner to a training dataset.
        """

        @override
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

    def __init__(self,
                 args: Namespace,
                 initial_state: ExperimentState,
                 dataset_splitter: DatasetSplitter,
                 training_procedure: Optional[TrainingProcedure] = None,
                 prediction_procedure: Optional[SkLearnExperiment.PredictionProcedure] = None):
        """
        :param args:                    The command line arguments specified by the user
        :param initial_state:           The initial state of the experiment
        :param dataset_splitter:        The method to be used for splitting the dataset into training and test datasets
        :param training_procedure:      The procedure that allows to fit a learner or None, if the default procedure
                                        should be used
        :param prediction_procedure:    The procedure that allows to obtain predictions from a learner or None, if the
                                        default procedure should be used
        """
        super().__init__(
            args=args,
            initial_state=initial_state,
            dataset_splitter=dataset_splitter,
            training_procedure=training_procedure if training_procedure else RuleLearnerExperiment.TrainingProcedure(
                base_learner=initial_state.problem_domain.base_learner,
                fit_kwargs=initial_state.problem_domain.fit_kwargs,
            ),
            prediction_procedure=prediction_procedure,
        )
