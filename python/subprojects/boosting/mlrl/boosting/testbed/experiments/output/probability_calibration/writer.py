"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow writing textual representations of probability calibration models to one or several sinks.
"""
import logging as log

from abc import ABC, abstractmethod
from typing import List, Optional, override

from mlrl.common.cython.probability_calibration import IsotonicProbabilityCalibrationModel, \
    NoProbabilityCalibrationModel
from mlrl.common.learners import ClassificationRuleLearner

from mlrl.boosting.testbed.experiments.output.probability_calibration.model_isotonic import IsotonicRegressionModel
from mlrl.boosting.testbed.experiments.output.probability_calibration.model_no import NoCalibrationModel

from mlrl.testbed.experiments.context import Context
from mlrl.testbed.experiments.output.data import OutputData
from mlrl.testbed.experiments.output.sinks import Sink
from mlrl.testbed.experiments.output.writer import DataExtractor, OutputWriter
from mlrl.testbed.experiments.state import ExperimentState


class ProbabilityCalibrationModelWriter(OutputWriter, ABC):
    """
    Allows writing textual representations of probability calibration models to one or several sinks.
    """

    class DefaultExtractor(DataExtractor, ABC):
        """
        An abstract base class for all classes that extract probability calibration models that are stored as part of a
        rule model.
        """

        @override
        def extract_data(self, state: ExperimentState, _: List[Sink]) -> Optional[OutputData]:
            """
            See :func:`mlrl.testbed.experiments.output.writer.DataExtractor.extract_data`
            """
            learner = state.learner_as(self, ClassificationRuleLearner)
            return self._get_calibration_model(learner) if learner else None

        @abstractmethod
        def _get_calibration_model(self, learner: ClassificationRuleLearner) -> Optional[OutputData]:
            """
            Must be implemented by subclasses in order to retrieve the calibration model from a rule learner.

            :param learner: The rule learner
            :return:        The calibration model
            """


class MarginalProbabilityCalibrationModelWriter(ProbabilityCalibrationModelWriter):
    """
    Allows writing textual representations of models for the calibration of marginal probabilities to one or several
    sinks.
    """

    class DefaultExtractor(ProbabilityCalibrationModelWriter.DefaultExtractor):
        """
        Extracts isotonic regression models for the calibration of marginal probabilities that are stores as part of a
        rule model.
        """

        @override
        def _get_calibration_model(self, learner: ClassificationRuleLearner) -> Optional[OutputData]:
            calibration_model = learner.marginal_probability_calibration_model_
            context = Context(include_dataset_type=False)
            properties = OutputData.Properties(name='Marginal probability calibration model',
                                               file_name='marginal_probability_calibration_model')

            if isinstance(calibration_model, IsotonicProbabilityCalibrationModel):
                return IsotonicRegressionModel(calibration_model=calibration_model,
                                               properties=properties,
                                               context=context,
                                               column_title_prefix='Label')

            if isinstance(calibration_model, NoProbabilityCalibrationModel):
                return NoCalibrationModel(properties=properties, context=context)

            log.error('%s expected type of calibration model to be %s, but calibration model has type %s',
                      type(self).__name__, IsotonicProbabilityCalibrationModel.__name__,
                      type(calibration_model).__name__)
            return None

    def __init__(self, *extractors: DataExtractor):
        """
        :param extractors: Extractors that should be used for extracting the output data to be written to the sinks
        """
        super().__init__(*extractors, MarginalProbabilityCalibrationModelWriter.DefaultExtractor())


class JointProbabilityCalibrationModelWriter(ProbabilityCalibrationModelWriter):
    """
    Allows writing textual representations of models for the calibration of joint probabilities to one or several sinks.
    """

    class DefaultExtractor(ProbabilityCalibrationModelWriter.DefaultExtractor):
        """
        Extracts isotonic regression models for the calibration of joint probabilities that are stores as part of a rule
        model.
        """

        @override
        def _get_calibration_model(self, learner: ClassificationRuleLearner) -> Optional[OutputData]:
            calibration_model = learner.joint_probability_calibration_model_
            context = Context(include_dataset_type=False)
            properties = OutputData.Properties(name='Joint probability calibration model',
                                               file_name='joint_probability_calibration_model')

            if isinstance(calibration_model, IsotonicProbabilityCalibrationModel):
                return IsotonicRegressionModel(calibration_model=calibration_model,
                                               properties=properties,
                                               context=context,
                                               column_title_prefix='Label vector')

            if isinstance(calibration_model, NoProbabilityCalibrationModel):
                return NoCalibrationModel(properties=properties, context=context)

            log.error('%s expected type of calibration model to be %s, but calibration model has type %s',
                      type(self).__name__, IsotonicProbabilityCalibrationModel.__name__,
                      type(calibration_model).__name__)
            return None

    def __init__(self, *extractors: DataExtractor):
        """
        :param extractors: Extractors that should be used for extracting the output data to be written to the sinks
        """
        super().__init__(*extractors, JointProbabilityCalibrationModelWriter.DefaultExtractor())
