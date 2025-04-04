"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for printing probability calibration models. The models can be written to one or several outputs, e.g.,
to the console or to a file.
"""
import logging as log

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from mlrl.common.config.options import Options
from mlrl.common.cython.probability_calibration import IsotonicProbabilityCalibrationModel, \
    IsotonicProbabilityCalibrationModelVisitor, NoProbabilityCalibrationModel
from mlrl.common.learners import ClassificationRuleLearner

from mlrl.testbed.experiments.output.data import OutputData, TabularOutputData
from mlrl.testbed.experiments.output.sinks import Sink
from mlrl.testbed.experiments.output.writer import OutputWriter
from mlrl.testbed.experiments.state import ExperimentState
from mlrl.testbed.util.format import OPTION_DECIMALS, format_number, format_table


class ProbabilityCalibrationModelWriter(OutputWriter, ABC):
    """
    An abstract base class for all classes that allow to write textual representations of probability calibration models
    to one or several sinks.
    """

    class IsotonicProbabilityCalibrationModelConverter(TabularOutputData, IsotonicProbabilityCalibrationModelVisitor):
        """
        Allows to create a textual representation of a model for the calibration of probabilities via isotonic
        regression.
        """

        def __init__(self,
                     calibration_model: IsotonicProbabilityCalibrationModel,
                     list_title: str,
                     name: str,
                     file_name: str,
                     formatter_options: ExperimentState.FormatterOptions = ExperimentState.FormatterOptions()):
            """
            :param calibration_model: The probability calibration model
            :param list_title:        The title of an individual list that is contained by the calibration model
            """
            super().__init__(name, file_name, formatter_options)
            self.calibration_model = calibration_model
            self.list_title = list_title
            self.bins: Dict[int, List[Tuple[float, float]]] = {}

        def __format_threshold_column(self, list_index: int) -> str:
            return self.list_title + ' ' + str(list_index + 1) + ' thresholds'

        def __format_probability_column(self, list_index: int) -> str:
            return self.list_title + ' ' + str(list_index + 1) + ' probabilities'

        def visit_bin(self, list_index: int, threshold: float, probability: float):
            """
            See :func:`mlrl.common.cython.probability_calibration.IsotonicProbabilityCalibrationModelVisitor.visit_bin`
            """
            bin_list = self.bins.setdefault(list_index, [])
            bin_list.append((threshold, probability))

        def to_text(self, options: Options, **_) -> Optional[str]:
            """
            See :func:`mlrl.testbed.experiments.output.data.OutputData.to_text`
            """
            self.calibration_model.visit(self)
            decimals = options.get_int(OPTION_DECIMALS, 4)
            bins = self.bins
            result = ''

            for list_index in sorted(bins.keys()):
                header = [self.__format_threshold_column(list_index), self.__format_probability_column(list_index)]
                rows = []

                for threshold, probability in bins[list_index]:
                    rows.append(
                        [format_number(threshold, decimals=decimals),
                         format_number(probability, decimals=decimals)])

                if result:
                    result += '\n'

                result += format_table(rows, header=header)

            return result

        def to_table(self, options: Options, **_) -> Optional[TabularOutputData.Table]:
            """
            See :func:`mlrl.testbed.experiments.output.data.TabularOutputData.to_table`
            """
            self.calibration_model.visit(self)
            decimals = options.get_int(OPTION_DECIMALS, 0)
            bins = self.bins
            rows = []
            end = False
            i = 0

            while not end:
                columns = {}
                end = True

                for list_index, bin_list in bins.items():
                    column_probability = self.__format_probability_column(list_index)
                    column_threshold = self.__format_threshold_column(list_index)

                    if len(bin_list) > i:
                        probability, threshold = bin_list[i]
                        columns[column_probability] = format_number(probability, decimals=decimals)
                        columns[column_threshold] = format_number(threshold, decimals=decimals)
                        end = False
                    else:
                        columns[column_probability] = None
                        columns[column_threshold] = None

                if not end:
                    rows.append(columns)

                i += 1

            return rows

    class NoProbabilityCalibrationModelConverter(TabularOutputData):
        """
        Allows to create a textual representation of a model for the calibration of probabilities that does not make any
        adjustments.
        """

        # pylint: disable=unused-argument
        def to_text(self, options: Options, **_) -> Optional[str]:
            """
            See :func:`mlrl.testbed.experiments.output.data.OutputData.to_text`
            """
            return 'No calibration model used'

        # pylint: disable=unused-argument
        def to_table(self, options: Options, **_) -> Optional[TabularOutputData.Table]:
            """
            See :func:`mlrl.testbed.experiments.output.data.TabularOutputData.to_table`
            """
            return None

    def __init__(self, name: str, file_name: str, formatter_options: ExperimentState.FormatterOptions, list_title: str,
                 *sinks: Sink):
        """
        :param list_title: The title of an individual list that is contained by a calibration model
        """
        super().__init__(*sinks)
        self.name = name
        self.file_name = file_name
        self.formatter_options = formatter_options
        self.list_title = list_title

    @abstractmethod
    def _get_calibration_model(self, learner: ClassificationRuleLearner) -> Any:
        """
        Must be implemented by subclasses in order to retrieve the calibration model from a rule learner.

        :param learner: The rule learner
        :return:        The calibration model
        """

    def _generate_output_data(self, state: ExperimentState) -> Optional[OutputData]:
        training_result = state.training_result

        if training_result:
            learner = training_result.learner
            if isinstance(learner, ClassificationRuleLearner):
                calibration_model = self._get_calibration_model(learner)

                if isinstance(calibration_model, IsotonicProbabilityCalibrationModel):
                    return ProbabilityCalibrationModelWriter.IsotonicProbabilityCalibrationModelConverter(
                        calibration_model, self.list_title, self.name, self.file_name, self.formatter_options)
                if isinstance(calibration_model, NoProbabilityCalibrationModel):
                    return ProbabilityCalibrationModelWriter.NoProbabilityCalibrationModelConverter(
                        self.name, self.file_name, self.formatter_options)

            log.error('The learner does not support to create a textual representation of the calibration model')

        return None


class MarginalProbabilityCalibrationModelWriter(ProbabilityCalibrationModelWriter):
    """
    Allow to write textual representations of models for the calibration of marginal probabilities to one or several
    sinks.
    """

    def __init__(self, *sinks: Sink):
        super().__init__('Marginal probability calibration model', 'marginal_probability_calibration_model',
                         ExperimentState.FormatterOptions(include_dataset_type=False), 'Label', *sinks)

    def _get_calibration_model(self, learner: ClassificationRuleLearner) -> Any:
        return learner.marginal_probability_calibration_model_


class JointProbabilityCalibrationModelWriter(ProbabilityCalibrationModelWriter):
    """
    Allow to write textual representations of models for the calibration of joint probabilities to one or several sinks.
    """

    def __init__(self, *sinks: Sink):
        super().__init__('Joint probability calibration model', 'joint_probability_calibration_model',
                         ExperimentState.FormatterOptions(include_dataset_type=False), 'Label vector', *sinks)

    def _get_calibration_model(self, learner: ClassificationRuleLearner) -> Any:
        return learner.joint_probability_calibration_model_
