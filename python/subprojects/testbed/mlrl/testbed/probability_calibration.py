"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for printing probability calibration models. The models can be written to one or several outputs, e.g.,
to the console or to a file.
"""
import logging as log

from abc import ABC, abstractmethod
from typing import Any, List, Optional

from mlrl.common.cython.probability_calibration import IsotonicProbabilityCalibrationModel, \
    NoProbabilityCalibrationModel
from mlrl.common.learners import ClassificationRuleLearner

from mlrl.testbed.experiments.output.data import OutputData
from mlrl.testbed.experiments.output.probability_calibration.model_isotonic import IsotonicRegressionModel
from mlrl.testbed.experiments.output.sinks import Sink
from mlrl.testbed.experiments.output.writer import DataExtractor, OutputWriter
from mlrl.testbed.experiments.state import ExperimentState


class ProbabilityCalibrationModelWriter(OutputWriter, ABC):
    """
    An abstract base class for all classes that allow to write textual representations of probability calibration models
    to one or several sinks.
    """

    class Extractor(DataExtractor, ABC):
        """
        An abstract base class for all classes that obtain probability calibration models from a learner.
        """

        def __init__(self, name: str, file_name: str, formatter_options: ExperimentState.FormatterOptions,
                     list_title: str):
            self.name = name
            self.file_name = file_name
            self.formatter_options = formatter_options
            self.list_title = list_title

        def extract_data(self, state: ExperimentState, _: List[Sink]) -> Optional[OutputData]:
            """
            See :func:`mlrl.testbed.experiments.output.writer.DataExtractor.extract_data`
            """
            training_result = state.training_result

            if training_result:
                learner = training_result.learner

                if isinstance(learner, ClassificationRuleLearner):
                    calibration_model = self._get_calibration_model(learner)

                    if isinstance(calibration_model, IsotonicProbabilityCalibrationModel):
                        return IsotonicRegressionModel(calibration_model=calibration_model,
                                                       name=self.name,
                                                       file_name=self.file_name,
                                                       default_formatter_options=self.formatter_options,
                                                       column_title_prefix=self.list_title)

                    if not isinstance(calibration_model, NoProbabilityCalibrationModel):
                        log.error('Cannot handle probability calibration model of type %s',
                                  type(calibration_model).__name__)

            return None

        @abstractmethod
        def _get_calibration_model(self, learner: ClassificationRuleLearner) -> Any:
            """
            Must be implemented by subclasses in order to retrieve the calibration model from a rule learner.

            :param learner: The rule learner
            :return:        The calibration model
            """


class MarginalProbabilityCalibrationModelWriter(ProbabilityCalibrationModelWriter):
    """
    Allow to write textual representations of models for the calibration of marginal probabilities to one or several
    sinks.
    """

    class Extractor(ProbabilityCalibrationModelWriter.Extractor):
        """
        Extracts models for the calibration of marginal probabilities from a learner.
        """

        def __init__(self):
            super().__init__(
                'Marginal probability calibration model',
                'marginal_probability_calibration_model',
                ExperimentState.FormatterOptions(include_dataset_type=False),
                'Label',
            )

        def _get_calibration_model(self, learner: ClassificationRuleLearner) -> Any:
            return learner.marginal_probability_calibration_model_

    def __init__(self, *extractors: DataExtractor):
        """
        :param extractors: Extractors that should be used for extracting the output data to be written to the sinks
        """
        super().__init__(*extractors, MarginalProbabilityCalibrationModelWriter.Extractor())


class JointProbabilityCalibrationModelWriter(ProbabilityCalibrationModelWriter):
    """
    Allow to write textual representations of models for the calibration of joint probabilities to one or several sinks.
    """

    class Extractor(ProbabilityCalibrationModelWriter.Extractor):
        """
        Extracts models for the calibration of joint probabilities from a learner.
        """

        def __init__(self):
            super().__init__(
                'Joint probability calibration model',
                'joint_probability_calibration_model',
                ExperimentState.FormatterOptions(include_dataset_type=False),
                'Label vector',
            )

        def _get_calibration_model(self, learner: ClassificationRuleLearner) -> Any:
            return learner.joint_probability_calibration_model_

    def __init__(self, *extractors: DataExtractor):
        """
        :param extractors: Extractors that should be used for extracting the output data to be written to the sinks
        """
        super().__init__(*extractors, JointProbabilityCalibrationModelWriter.Extractor())
