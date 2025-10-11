"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow writing textual representations of probability calibration models to one or several sinks.
"""
import logging as log

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, override

from mlrl.common.cython.probability_calibration import IsotonicProbabilityCalibrationModel, \
    NoProbabilityCalibrationModel
from mlrl.common.learners import ClassificationRuleLearner

from mlrl.boosting.testbed.experiments.output.probability_calibration.model_isotonic import IsotonicRegressionModel
from mlrl.boosting.testbed.experiments.output.probability_calibration.model_no import NoCalibrationModel

from mlrl.testbed.experiments.context import Context
from mlrl.testbed.experiments.data import TabularProperties
from mlrl.testbed.experiments.input.data import TabularInputData
from mlrl.testbed.experiments.output.data import OutputData
from mlrl.testbed.experiments.output.sinks import Sink
from mlrl.testbed.experiments.output.writer import DataExtractor, ResultWriter, TabularDataExtractor
from mlrl.testbed.experiments.state import ExperimentState
from mlrl.testbed.experiments.table import ColumnWiseTable

from mlrl.util.options import Options


class ProbabilityCalibrationModelWriter(ResultWriter, ABC):
    """
    Allows writing textual representations of probability calibration models to one or several sinks.
    """

    class InputExtractor(TabularDataExtractor, ABC):
        """
        An abstract base class for all classes that use `TabularInputData` that has previously been loaded via an input
        reader.
        """

        def __create_isotonic_calibration_model(self, table: ColumnWiseTable) -> Optional[IsotonicRegressionModel]:
            bin_lists: Dict[int, IsotonicRegressionModel.BinList] = {}

            for column in table.columns:
                header = column.header

                if header:
                    parts = header.split()
                    column_type = parts[-1]
                    list_index = int(parts[-2]) - 1
                    bin_list = bin_lists.setdefault(list_index, IsotonicRegressionModel.BinList())

                    if column_type == IsotonicRegressionModel.COLUMN_PROBABILITIES:
                        for probability in column:
                            if probability:
                                bin_list.probabilities.append(float(probability))
                    elif column_type == IsotonicRegressionModel.COLUMN_THRESHOLDS:
                        for threshold in column:
                            if threshold:
                                bin_list.thresholds.append(float(threshold))

            return self._create_isotonic_calibration_model(bin_lists) if bin_lists else None

        @override
        def extract_data(self, state: ExperimentState, sinks: List[Sink]) -> List[Tuple[ExperimentState, OutputData]]:
            """
            See :func:`mlrl.testbed.experiments.output.writer.DataExtractor.extract_data`
            """
            result: List[Tuple[ExperimentState, OutputData]] = []

            for extracted_state, tabular_output_data in super().extract_data(state, sinks):
                table = tabular_output_data.to_table(Options()).to_column_wise_table()
                result.append((extracted_state, self.__create_isotonic_calibration_model(table)))

            return result

        @abstractmethod
        def _create_isotonic_calibration_model(
                self, bin_lists: Dict[int, IsotonicRegressionModel.BinList]) -> IsotonicRegressionModel:
            """
            Must be implemented by subclasses in order to create objects of type `IsotonicRegressionModel`.

            :param bin_lists:   A dictionary that stores lists of bins contained in an isotonic regression model, mapped
                                to indices
            :return:            An object of type `IsotonicRegressionModel` that has been created
            """

    class DefaultExtractor(DataExtractor, ABC):
        """
        An abstract base class for all classes that extract probability calibration models that are stored as part of a
        rule model.
        """

        @override
        def extract_data(self, state: ExperimentState, _: List[Sink]) -> List[Tuple[ExperimentState, OutputData]]:
            """
            See :func:`mlrl.testbed.experiments.output.writer.DataExtractor.extract_data`
            """
            learner = state.learner_as(self, ClassificationRuleLearner)

            if learner:
                return [(state, self._get_calibration_model(learner))]

            return []

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

    PROPERTIES = TabularProperties(name='Marginal probability calibration model',
                                   file_name='marginal_probability_calibration_model')

    CONTEXT = Context(include_dataset_type=False)

    class InputExtractor(ProbabilityCalibrationModelWriter.InputExtractor):
        """
        Uses `TabularInputData` that has previously been loaded via an input reader.
        """

        @override
        def _create_isotonic_calibration_model(
                self, bin_lists: Dict[int, IsotonicRegressionModel.BinList]) -> IsotonicRegressionModel:
            return IsotonicRegressionModel(bin_lists=bin_lists,
                                           properties=MarginalProbabilityCalibrationModelWriter.PROPERTIES,
                                           context=MarginalProbabilityCalibrationModelWriter.CONTEXT,
                                           column_title_prefix='Label')

    class DefaultExtractor(ProbabilityCalibrationModelWriter.DefaultExtractor):
        """
        Extracts isotonic regression models for the calibration of marginal probabilities that are stores as part of a
        rule model.
        """

        @override
        def _get_calibration_model(self, learner: ClassificationRuleLearner) -> Optional[OutputData]:
            calibration_model = learner.marginal_probability_calibration_model_

            if isinstance(calibration_model, IsotonicProbabilityCalibrationModel):
                return IsotonicRegressionModel.from_calibration_model(
                    calibration_model=calibration_model,
                    properties=MarginalProbabilityCalibrationModelWriter.PROPERTIES,
                    context=MarginalProbabilityCalibrationModelWriter.CONTEXT,
                    column_title_prefix='Label')

            if isinstance(calibration_model, NoProbabilityCalibrationModel):
                return NoCalibrationModel(properties=MarginalProbabilityCalibrationModelWriter.PROPERTIES,
                                          context=MarginalProbabilityCalibrationModelWriter.CONTEXT)

            log.error('%s expected type of calibration model to be %s, but calibration model has type %s',
                      type(self).__name__, IsotonicProbabilityCalibrationModel.__name__,
                      type(calibration_model).__name__)
            return None

    def __init__(self, *extractors: DataExtractor):
        """
        :param extractors: Extractors that should be used for extracting the output data to be written to the sinks
        """
        super().__init__(MarginalProbabilityCalibrationModelWriter.InputExtractor(properties=self.PROPERTIES,
                                                                                  context=self.CONTEXT),
                         *extractors,
                         MarginalProbabilityCalibrationModelWriter.DefaultExtractor(),
                         input_data=TabularInputData(properties=self.PROPERTIES, context=self.CONTEXT))


class JointProbabilityCalibrationModelWriter(ProbabilityCalibrationModelWriter):
    """
    Allows writing textual representations of models for the calibration of joint probabilities to one or several sinks.
    """

    PROPERTIES = TabularProperties(name='Joint probability calibration model',
                                   file_name='joint_probability_calibration_model')

    CONTEXT = Context(include_dataset_type=False)

    class InputExtractor(ProbabilityCalibrationModelWriter.InputExtractor):
        """
        Uses `TabularInputData` that has previously been loaded via an input reader.
        """

        @override
        def _create_isotonic_calibration_model(
                self, bin_lists: Dict[int, IsotonicRegressionModel.BinList]) -> IsotonicRegressionModel:
            return IsotonicRegressionModel(bin_lists=bin_lists,
                                           properties=JointProbabilityCalibrationModelWriter.PROPERTIES,
                                           context=JointProbabilityCalibrationModelWriter.CONTEXT,
                                           column_title_prefix='Label vector')

    class DefaultExtractor(ProbabilityCalibrationModelWriter.DefaultExtractor):
        """
        Extracts isotonic regression models for the calibration of joint probabilities that are stores as part of a rule
        model.
        """

        @override
        def _get_calibration_model(self, learner: ClassificationRuleLearner) -> Optional[OutputData]:
            calibration_model = learner.joint_probability_calibration_model_

            if isinstance(calibration_model, IsotonicProbabilityCalibrationModel):
                return IsotonicRegressionModel.from_calibration_model(
                    calibration_model=calibration_model,
                    properties=JointProbabilityCalibrationModelWriter.PROPERTIES,
                    context=JointProbabilityCalibrationModelWriter.CONTEXT,
                    column_title_prefix='Label vector')

            if isinstance(calibration_model, NoProbabilityCalibrationModel):
                return NoCalibrationModel(properties=JointProbabilityCalibrationModelWriter.PROPERTIES,
                                          context=JointProbabilityCalibrationModelWriter.CONTEXT)

            log.error('%s expected type of calibration model to be %s, but calibration model has type %s',
                      type(self).__name__, IsotonicProbabilityCalibrationModel.__name__,
                      type(calibration_model).__name__)
            return None

    def __init__(self, *extractors: DataExtractor):
        """
        :param extractors: Extractors that should be used for extracting the output data to be written to the sinks
        """
        super().__init__(JointProbabilityCalibrationModelWriter.InputExtractor(properties=self.PROPERTIES,
                                                                               context=self.CONTEXT),
                         *extractors,
                         JointProbabilityCalibrationModelWriter.DefaultExtractor(),
                         input_data=TabularInputData(properties=self.PROPERTIES, context=self.CONTEXT))
