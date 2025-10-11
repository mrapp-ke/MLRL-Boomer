"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow writing characteristics of models to one or several sinks.
"""
import logging as log

from typing import List, Optional, override

import numpy as np

from mlrl.common.cython.rule_model import Body, CompleteHead, ConjunctiveBody, EmptyBody, Head, PartialHead, RuleModel
from mlrl.common.mixins import ClassifierMixin, RegressorMixin
from mlrl.common.testbed.experiments.output.characteristics.model.characteristics import BodyStatistics, \
    HeadStatistics, RuleModelCharacteristics, RuleModelStatistics, RuleStatistics

from mlrl.testbed.experiments.input.data import TabularInputData
from mlrl.testbed.experiments.output.data import OutputData
from mlrl.testbed.experiments.output.sinks import Sink
from mlrl.testbed.experiments.output.writer import DataExtractor, ResultWriter, TabularDataExtractor
from mlrl.testbed.experiments.state import ExperimentState

from mlrl.util.options import Options


class RuleModelCharacteristicsWriter(ResultWriter):
    """
    Allows writing the characteristics of a model to one or several sinks.
    """

    class InputExtractor(TabularDataExtractor):
        """
        Uses `TabularInputData` that has previously been loaded via an input reader.
        """

        @override
        def extract_data(self, state: ExperimentState, sinks: List[Sink]) -> Optional[OutputData]:
            """
            See :func:`mlrl.testbed.experiments.output.writer.DataExtractor.extract_data`
            """
            tabular_output_data = super().extract_data(state, sinks)

            if tabular_output_data:
                table = tabular_output_data.to_table(Options())
                table.to_column_wise_table()
                indices = {column.header: index for index, column in enumerate(table.columns)}
                row_wise_table = table.to_row_wise_table()
                rule_model_statistics = RuleModelStatistics()

                for row in row_wise_table.rows:
                    body_statistics = BodyStatistics(
                        num_numerical_leq=int(
                            row[indices[RuleModelCharacteristics.COLUMN_NUM_CONDITIONS_NUMERICAL_LEQ]]),
                        num_numerical_gr=int(row[indices[RuleModelCharacteristics.COLUMN_NUM_CONDITIONS_NUMERICAL_GR]]),
                        num_ordinal_leq=int(row[indices[RuleModelCharacteristics.COLUMN_NUM_CONDITIONS_ORDINAL_LEQ]]),
                        num_ordinal_gr=int(row[indices[RuleModelCharacteristics.COLUMN_NUM_CONDITIONS_ORDINAL_GR]]),
                        num_nominal_eq=int(row[indices[RuleModelCharacteristics.COLUMN_NUM_CONDITIONS_NOMINAL_EQ]]),
                        num_nominal_neq=int(row[indices[RuleModelCharacteristics.COLUMN_NUM_CONDITIONS_NOMINAL_NEQ]]),
                    )
                    head_statistics = HeadStatistics(
                        num_positive_predictions=int(
                            row[indices[RuleModelCharacteristics.COLUMN_NUM_PREDICTIONS_POSITIVE]]),
                        num_negative_predictions=int(
                            row[indices[RuleModelCharacteristics.COLUMN_NUM_PREDICTIONS_NEGATIVE]]),
                    )
                    rule_statistics = RuleStatistics(body_statistics=body_statistics, head_statistics=head_statistics)

                    if row[indices[RuleModelCharacteristics.COLUMN_INDEX]].lower().find('default') > 0:
                        rule_model_statistics.default_rule_statistics = rule_statistics
                    else:
                        rule_model_statistics.rule_statistics.append(rule_statistics)

                return RuleModelCharacteristics(rule_model_statistics)

            return None

    class DefaultExtractor(DataExtractor):
        """
        Allows to extract characteristics from a `RuleModel`.
        """

        def __create_rule_model_characteristics(self, model: RuleModel) -> RuleModelCharacteristics:
            statistics = RuleModelStatistics()

            for rule in model:
                default_rule = self.__create_body_characteristics(statistics, rule.body)
                self.__create_head_characteristics(statistics, rule.head, default_rule=default_rule)

            return RuleModelCharacteristics(statistics)

        @staticmethod
        def __create_body_characteristics(statistics: RuleModelStatistics, body: Body) -> bool:
            if isinstance(body, EmptyBody):
                statistics.default_rule_statistics = RuleStatistics(body_statistics=BodyStatistics())
                return True

            if isinstance(body, ConjunctiveBody):
                body_statistics = BodyStatistics(
                    num_numerical_leq=body.numerical_leq_indices.size if body.numerical_leq_indices is not None else 0,
                    num_numerical_gr=body.numerical_gr_indices.size if body.numerical_gr_indices is not None else 0,
                    num_ordinal_leq=body.ordinal_leq_indices.size if body.ordinal_leq_indices is not None else 0,
                    num_ordinal_gr=body.ordinal_gr_indices.size if body.ordinal_gr_indices is not None else 0,
                    num_nominal_eq=body.nominal_eq_indices.size if body.nominal_eq_indices is not None else 0,
                    num_nominal_neq=body.nominal_neq_indices.size if body.nominal_neq_indices is not None else 0)
                statistics.rule_statistics.append(RuleStatistics(body_statistics=body_statistics))
                return False

            raise ValueError('Unsupported type of body: ' + str(type(body)))

        @staticmethod
        def __create_head_characteristics(statistics: RuleModelStatistics, head: Head, default_rule: bool):
            head_statistics = None

            if isinstance(head, CompleteHead):
                num_positive_predictions = int(np.count_nonzero(head.scores > 0))
                num_negative_predictions = int(head.scores.shape[0] - num_positive_predictions)
                head_statistics = HeadStatistics(num_positive_predictions=num_positive_predictions,
                                                 num_negative_predictions=num_negative_predictions)
            elif isinstance(head, PartialHead):
                num_positive_predictions = int(np.count_nonzero(head.scores > 0))
                num_negative_predictions = int(head.scores.shape[0] - num_positive_predictions)
                head_statistics = HeadStatistics(num_positive_predictions=num_positive_predictions,
                                                 num_negative_predictions=num_negative_predictions)

            if head_statistics:
                if default_rule:
                    default_rule_statistics = statistics.default_rule_statistics

                    if default_rule_statistics:
                        default_rule_statistics.head_statistics = head_statistics
                else:
                    statistics.rule_statistics[-1].head_statistics = head_statistics
            else:
                raise ValueError('Unsupported type of head: ' + str(type(head)))

        @override
        def extract_data(self, state: ExperimentState, _: List[Sink]) -> Optional[OutputData]:
            """
            See :func:`mlrl.testbed.experiments.output.writer.DataExtractor.extract_data`
            """
            learner = state.learner_as(self, ClassifierMixin, RegressorMixin)

            if learner:
                model = learner.model_

                if isinstance(model, RuleModel):
                    return self.__create_rule_model_characteristics(model)

                log.error('%s expected type of model to be %s, but model has type %s',
                          type(self).__name__, RuleModel.__name__,
                          type(model).__name__)

            return None

    def __init__(self, *extractors: DataExtractor):
        """
        :param extractors: Extractors that should be used for extracting the output data to be written to the sinks
        """
        super().__init__(RuleModelCharacteristicsWriter.InputExtractor(properties=RuleModelCharacteristics.PROPERTIES,
                                                                       context=RuleModelCharacteristics.CONTEXT),
                         *extractors,
                         RuleModelCharacteristicsWriter.DefaultExtractor(),
                         input_data=TabularInputData(properties=RuleModelCharacteristics.PROPERTIES,
                                                     context=RuleModelCharacteristics.CONTEXT))
