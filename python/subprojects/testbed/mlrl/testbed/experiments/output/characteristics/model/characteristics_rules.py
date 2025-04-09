"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for representing characteristics of rule models that are part of output data.
"""

from functools import reduce
from itertools import chain
from typing import List, Optional

from mlrl.common.config.options import Options

from mlrl.testbed.experiments.output.characteristics.model.statistics_rules import BodyStatistics, HeadStatistics, \
    RuleModelStatistics, RuleStatistics
from mlrl.testbed.experiments.output.data import TabularOutputData
from mlrl.testbed.experiments.output.table import RowWiseTable, Table
from mlrl.testbed.experiments.state import ExperimentState
from mlrl.testbed.util.format import format_number, format_percentage, format_table
from mlrl.testbed.util.math import divide_or_zero


class RuleModelCharacteristics(TabularOutputData):
    """
    Represents characteristics of a rule model that are part of output data.
    """

    def __init__(self, statistics: RuleModelStatistics):
        """
        :param statistics: The statistics of a rule model
        """
        super().__init__(name='Model characteristics',
                         file_name='model_characteristics',
                         default_formatter_options=ExperimentState.FormatterOptions(include_dataset_type=False))
        self.statistics = statistics

    # pylint: disable=unused-argument
    def to_text(self, options: Options, **_) -> Optional[str]:
        """
        See :func:`mlrl.testbed.experiments.output.data.OutputData.to_text`
        """
        aggregated_rule_statistics = reduce(lambda aggr, rule_statistics: aggr + rule_statistics,
                                            self.statistics.rule_statistics, RuleStatistics())
        text = self.__format_aggregated_body_statistics(aggregated_rule_statistics) + '\n\n'
        text += self.__format_aggregated_head_statistics(aggregated_rule_statistics) + '\n\n'
        text += self.__format_aggregated_rule_statistics(aggregated_rule_statistics)
        return text

    @staticmethod
    def __format_body_statistics(body_statistics: BodyStatistics) -> List[str]:
        return [
            str(body_statistics.num_conditions),
            format_percentage(body_statistics.fraction_numerical_leq),
            format_percentage(body_statistics.fraction_numerical_gr),
            format_percentage(body_statistics.fraction_ordinal_leq),
            format_percentage(body_statistics.fraction_ordinal_gr),
            format_percentage(body_statistics.fraction_nominal_eq),
            format_percentage(body_statistics.fraction_nominal_neq),
        ]

    def __format_aggregated_body_statistics(self, aggregated_rule_statistics: RuleStatistics) -> str:
        statistics = self.statistics
        rows = []

        if statistics.has_default_rule:
            rows.append(['Default rule']
                        + self.__format_body_statistics(statistics.default_rule_statistics.body_statistics))

        rows.append([str(statistics.num_rules) + ' local rules']
                    + self.__format_body_statistics(aggregated_rule_statistics.body_statistics))

        return format_table(rows,
                            header=[
                                'Statistics about conditions', 'Total', 'Numerical <= operator', 'Numerical > operator',
                                'Ordinal <= operator', 'Ordinal > operator', 'Nominal == operator',
                                'Nominal != operator'
                            ],
                            alignment=['left', 'right', 'right', 'right', 'right', 'right', 'right', 'right'])

    @staticmethod
    def __format_head_statistics(head_statistics: HeadStatistics) -> List[str]:
        return [
            str(head_statistics.num_predictions),
            format_percentage(head_statistics.fraction_positive_predictions),
            format_percentage(head_statistics.fraction_negative_predictions),
        ]

    def __format_aggregated_head_statistics(self, aggregated_rule_statistics: RuleStatistics) -> str:
        statistics = self.statistics
        rows = []

        if statistics.has_default_rule:
            rows.append(['Default rule']
                        + self.__format_head_statistics(statistics.default_rule_statistics.head_statistics))

        rows.append([str(statistics.num_rules) + ' local rules']
                    + self.__format_head_statistics(aggregated_rule_statistics.head_statistics))

        return format_table(rows,
                            header=['Statistics about predictions', 'Total', 'Positive', 'Negative'],
                            alignment=['left', 'right', 'right', 'right'])

    def __format_aggregated_rule_statistics(self, aggregated_rule_statistics: RuleStatistics) -> str:
        statistics = self.statistics
        return format_table(
            rows=[
                [
                    'Conditions',
                    format_number(statistics.min_conditions),
                    format_number(
                        divide_or_zero(aggregated_rule_statistics.body_statistics.num_conditions,
                                       statistics.num_rules)),
                    format_number(statistics.max_conditions)
                ],
                [
                    'Predictions',
                    format_number(statistics.min_predictions),
                    format_number(
                        divide_or_zero(aggregated_rule_statistics.head_statistics.num_predictions,
                                       statistics.num_rules)),
                    format_number(statistics.max_predictions)
                ],
            ],
            header=['Statistics per local rule', 'Minimum', 'Average', 'Maximum'],
        )

    # pylint: disable=unused-argument
    def to_table(self, options: Options, **_) -> Optional[Table]:
        """
        See :func:`mlrl.testbed.experiments.output.data.TabularOutputData.to_table`
        """
        statistics = self.statistics
        has_default_rule = statistics.has_default_rule
        default_rule_statistics = [statistics.default_rule_statistics] if has_default_rule else []
        table = RowWiseTable.empty('Rule', 'conditions', 'numerical conditions', 'numerical <= operator',
                                   'numerical > operator', 'ordinal conditions', 'ordinal <= operator',
                                   'ordinal > operator', 'nominal conditions', 'nominal == operator',
                                   'nominal != operator', 'predictions', 'pos. predictions', 'neg. predictions')

        for i, rule_statistics in enumerate(chain(default_rule_statistics, statistics.rule_statistics)):
            rule_name = 'Rule ' + str(i + 1)

            if i == 0 and has_default_rule:
                rule_name += ' (Default rule)'

            body_statistics = rule_statistics.body_statistics
            head_statistics = rule_statistics.head_statistics
            table.add_row(rule_name, body_statistics.num_conditions, body_statistics.num_numerical,
                          body_statistics.num_numerical_leq, body_statistics.num_numerical_gr,
                          body_statistics.num_ordinal, body_statistics.num_ordinal_leq, body_statistics.num_ordinal_gr,
                          body_statistics.num_nominal, body_statistics.num_nominal_eq, body_statistics.num_nominal_neq,
                          head_statistics.num_predictions, head_statistics.num_positive_predictions,
                          head_statistics.num_negative_predictions)

        return table
