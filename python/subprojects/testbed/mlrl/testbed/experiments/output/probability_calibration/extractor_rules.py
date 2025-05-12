"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for extracting probability calibration models that are stored as part of rule models.
"""
import logging as log

from abc import ABC, abstractmethod
from typing import Any, List, Optional

from mlrl.common.cython.probability_calibration import IsotonicProbabilityCalibrationModel, \
    NoProbabilityCalibrationModel
from mlrl.common.learners import ClassificationRuleLearner

from mlrl.testbed.experiments.context import Context
from mlrl.testbed.experiments.output.data import OutputData
from mlrl.testbed.experiments.output.probability_calibration.model_isotonic import IsotonicRegressionModel
from mlrl.testbed.experiments.output.sinks import Sink
from mlrl.testbed.experiments.output.writer import DataExtractor
from mlrl.testbed.experiments.state import ExperimentState


class ProbabilityCalibrationModelExtractor(DataExtractor, ABC):
    """
    An abstract base class for all classes that extract probability calibration models that are stored as part of a rule
    model.
    """

    def extract_data(self, state: ExperimentState, _: List[Sink]) -> Optional[OutputData]:
        """
        See :func:`mlrl.testbed.experiments.output.writer.DataExtractor.extract_data`
        """
        training_result = state.training_result

        if training_result:
            learner = training_result.learner

            if isinstance(learner, ClassificationRuleLearner):
                return self._get_calibration_model(learner)

        return None

    @abstractmethod
    def _get_calibration_model(self, learner: ClassificationRuleLearner) -> Optional[OutputData]:
        """
        Must be implemented by subclasses in order to retrieve the calibration model from a rule learner.

        :param learner: The rule learner
        :return:        The calibration model
        """


class IsotonicMarginalProbabilityCalibrationModelExtractor(ProbabilityCalibrationModelExtractor):
    """
    Extracts isotonic regression models for the calibration of marginal probabilities that are stores as part of a rule
    model.
    """

    def _get_calibration_model(self, learner: ClassificationRuleLearner) -> Any:
        calibration_model = learner.marginal_probability_calibration_model_

        if isinstance(calibration_model, IsotonicProbabilityCalibrationModel):
            return IsotonicRegressionModel(calibration_model=calibration_model,
                                           properties=OutputData.Properties(
                                               name='Marginal probability calibration model',
                                               file_name='marginal_probability_calibration_model',
                                           ),
                                           context=Context(include_dataset_type=False),
                                           column_title_prefix='Label')

        if not isinstance(calibration_model, NoProbabilityCalibrationModel):
            log.error('Cannot handle probability calibration model of type %s', type(calibration_model).__name__)

        return calibration_model


class IsotonicJointProbabilityCalibrationModelExtractor(ProbabilityCalibrationModelExtractor):
    """
    Extracts isotonic regression models for the calibration of joint probabilities that are stores as part of a rule
    model.
    """

    def _get_calibration_model(self, learner: ClassificationRuleLearner) -> Any:
        calibration_model = learner.joint_probability_calibration_model_

        if isinstance(calibration_model, IsotonicProbabilityCalibrationModel):
            return IsotonicRegressionModel(calibration_model=calibration_model,
                                           properties=OutputData.Properties(
                                               name='Joint probability calibration model',
                                               file_name='joint_probability_calibration_model',
                                           ),
                                           context=Context(include_dataset_type=False),
                                           column_title_prefix='Label vector')

        if not isinstance(calibration_model, NoProbabilityCalibrationModel):
            log.error('Cannot handle probability calibration model of type %s', type(calibration_model).__name__)

        return calibration_model
