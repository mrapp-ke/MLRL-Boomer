"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for representing characteristics of rule models that are part of output data.
"""

from functools import reduce
from itertools import chain
from typing import override
from rich.console import ConsoleRenderable, Group
from rich.text import Text

from mlrl.common.testbed.experiments.output.characteristics.model.statistics import (
    BodyStatistics,
    HeadStatistics,
    RuleModelStatistics,
    RuleStatistics,
)

from mlrl.testbed.experiments.context import Context
from mlrl.testbed.experiments.data import TabularProperties
from mlrl.testbed.experiments.output.data import TabularOutputData
from mlrl.testbed.experiments.table import Column, RowWiseTable, Table
from mlrl.testbed.util.format import format_percentage
from mlrl.testbed.util.math import divide_or_zero

from mlrl.util.format import format_value
from mlrl.util.options import Options


class RuleModelCharacteristics(TabularOutputData):
    """
    Represents characteristics of a rule model that are part of output data.
    """

    PROPERTIES = TabularProperties(name='Model characteristics', file_name='model_characteristics')

    CONTEXT = Context(include_dataset_type=False)

    COLUMN_INDEX = 'Rule'

    COLUMN_NUM_CONDITIONS = 'conditions'

    COLUMN_NUM_CONDITIONS_NUMERICAL = 'numerical conditions'

    COLUMN_NUM_CONDITIONS_NUMERICAL_LEQ = 'numerical <= operator'

    COLUMN_NUM_CONDITIONS_NUMERICAL_GR = 'numerical > operator'

    COLUMN_NUM_CONDITIONS_ORDINAL = 'ordinal conditions'

    COLUMN_NUM_CONDITIONS_ORDINAL_LEQ = 'ordinal <= operator'

    COLUMN_NUM_CONDITIONS_ORDINAL_GR = 'ordinal > operator'

    COLUMN_NUM_CONDITION_NOMINAL = 'nominal conditions'

    COLUMN_NUM_CONDITIONS_NOMINAL_EQ = 'nominal == operator'

    COLUMN_NUM_CONDITIONS_NOMINAL_NEQ = 'nominal != operator'

    COLUMN_NUM_PREDICTIONS = 'predictions'

    COLUMN_NUM_PREDICTIONS_POSITIVE = 'pos. predictions'

    COLUMN_NUM_PREDICTIONS_NEGATIVE = 'neg. predictions'

    def __init__(self, statistics: RuleModelStatistics):
        """
        :param statistics: The statistics of a rule model
        """
        super().__init__(properties=self.PROPERTIES, context=self.CONTEXT)
        self.statistics = statistics

    @override
    def to_text(self, options: Options, **_) -> str | ConsoleRenderable | None:
        """
        See :func:`mlrl.testbed.experiments.output.data.TextualOutputData.to_text`
        """
        aggregated_rule_statistics = reduce(
            lambda aggr, rule_statistics: aggr + rule_statistics, self.statistics.rule_statistics, RuleStatistics()
        )
        return Group(
            self.__format_aggregated_body_statistics(aggregated_rule_statistics),
            Text(''),
            self.__format_aggregated_head_statistics(aggregated_rule_statistics),
            Text(''),
            self.__format_aggregated_rule_statistics(aggregated_rule_statistics),
        )

    @staticmethod
    def __format_body_statistics(body_statistics: BodyStatistics) -> list[str]:
        return [
            str(body_statistics.num_conditions),
            format_percentage(body_statistics.fraction_numerical_leq),
            format_percentage(body_statistics.fraction_numerical_gr),
            format_percentage(body_statistics.fraction_ordinal_leq),
            format_percentage(body_statistics.fraction_ordinal_gr),
            format_percentage(body_statistics.fraction_nominal_eq),
            format_percentage(body_statistics.fraction_nominal_neq),
        ]

    def __format_aggregated_body_statistics(self, aggregated_rule_statistics: RuleStatistics) -> ConsoleRenderable:
        statistics = self.statistics
        headers = [
            'Statistics about conditions',
            'Total',
            'Numerical <= operator',
            'Numerical > operator',
            'Ordinal <= operator',
            'Ordinal > operator',
            'Nominal == operator',
            'Nominal != operator',
        ]
        table = RowWiseTable(*headers)
        default_rule_statistics = statistics.default_rule_statistics

        if default_rule_statistics:
            body_statistics = default_rule_statistics.body_statistics
            table.add_row('Default rule', *self.__format_body_statistics(body_statistics))

        body_statistics = aggregated_rule_statistics.body_statistics
        table.add_row(f'{statistics.num_rules} local rules', *self.__format_body_statistics(body_statistics))
        column_styles = [Column.Style.HEADER] + [Column.Style.VALUE for _ in range(len(headers) - 1)]
        column_alignments = [Column.Alignment.LEFT] + [Column.Alignment.RIGHT for _ in range(len(headers) - 1)]
        return table.to_rich_table(
            auto_rotate=False,
            border_style=Table.BorderStyle.INNER_LINES,
            column_styles=column_styles,
            column_alignments=column_alignments,
        )

    @staticmethod
    def __format_head_statistics(head_statistics: HeadStatistics) -> list[str]:
        return [
            str(head_statistics.num_predictions),
            format_percentage(head_statistics.fraction_positive_predictions),
            format_percentage(head_statistics.fraction_negative_predictions),
        ]

    def __format_aggregated_head_statistics(self, aggregated_rule_statistics: RuleStatistics) -> ConsoleRenderable:
        statistics = self.statistics
        headers = ['Statistics about predictions', 'Total', 'Positive', 'Negative']
        table = RowWiseTable(*headers)
        default_rule_statistics = statistics.default_rule_statistics

        if default_rule_statistics:
            head_statistics = default_rule_statistics.head_statistics
            table.add_row('Default rule', *self.__format_head_statistics(head_statistics))

        head_statistics = aggregated_rule_statistics.head_statistics
        table.add_row(f'{statistics.num_rules} local rules', *self.__format_head_statistics(head_statistics))
        column_styles = [Column.Style.HEADER] + [Column.Style.VALUE for _ in range(len(headers) - 1)]
        column_alignments = [Column.Alignment.LEFT] + [Column.Alignment.RIGHT for _ in range(len(headers) - 1)]
        return table.to_rich_table(
            auto_rotate=False,
            border_style=Table.BorderStyle.INNER_LINES,
            column_styles=column_styles,
            column_alignments=column_alignments,
        )

    def __format_aggregated_rule_statistics(self, aggregated_rule_statistics: RuleStatistics) -> ConsoleRenderable:
        statistics = self.statistics
        num_rules = statistics.num_rules
        headers = ['Statistics per local rule', 'Minimum', 'Average', 'Maximum']
        table = RowWiseTable(*headers)
        table.add_row(
            'Conditions',
            format_value(statistics.min_conditions),
            format_value(divide_or_zero(aggregated_rule_statistics.body_statistics.num_conditions, num_rules)),
            format_value(statistics.max_conditions),
        )
        table.add_row(
            'Predictions',
            format_value(statistics.min_predictions),
            format_value(divide_or_zero(aggregated_rule_statistics.head_statistics.num_predictions, num_rules)),
            format_value(statistics.max_predictions),
        )
        column_styles = [Column.Style.HEADER] + [Column.Style.VALUE for _ in range(len(headers) - 1)]
        column_alignments = [Column.Alignment.LEFT] + [Column.Alignment.RIGHT for _ in range(len(headers) - 1)]
        return table.to_rich_table(
            auto_rotate=False,
            border_style=Table.BorderStyle.INNER_LINES,
            column_styles=column_styles,
            column_alignments=column_alignments,
        )

    @override
    def to_table(self, options: Options, **_) -> Table | None:
        """
        See :func:`mlrl.testbed.experiments.output.data.TabularOutputData.to_table`
        """
        statistics = self.statistics
        default_rule_statistics = [statistics.default_rule_statistics] if statistics.default_rule_statistics else []
        table = RowWiseTable(
            self.COLUMN_INDEX,
            self.COLUMN_NUM_CONDITIONS,
            self.COLUMN_NUM_CONDITIONS_NUMERICAL,
            self.COLUMN_NUM_CONDITIONS_NUMERICAL_LEQ,
            self.COLUMN_NUM_CONDITIONS_NUMERICAL_GR,
            self.COLUMN_NUM_CONDITIONS_ORDINAL,
            self.COLUMN_NUM_CONDITIONS_ORDINAL_LEQ,
            self.COLUMN_NUM_CONDITIONS_ORDINAL_GR,
            self.COLUMN_NUM_CONDITION_NOMINAL,
            self.COLUMN_NUM_CONDITIONS_NOMINAL_EQ,
            self.COLUMN_NUM_CONDITIONS_NOMINAL_NEQ,
            self.COLUMN_NUM_PREDICTIONS,
            self.COLUMN_NUM_PREDICTIONS_POSITIVE,
            self.COLUMN_NUM_PREDICTIONS_NEGATIVE,
        )

        for i, rule_statistics in enumerate(chain(default_rule_statistics, statistics.rule_statistics)):
            rule_name = f'Rule {i + 1}'

            if i == 0 and statistics.has_default_rule:
                rule_name += ' (Default rule)'

            body_statistics = rule_statistics.body_statistics
            head_statistics = rule_statistics.head_statistics
            table.add_row(
                rule_name,
                body_statistics.num_conditions,
                body_statistics.num_numerical,
                body_statistics.num_numerical_leq,
                body_statistics.num_numerical_gr,
                body_statistics.num_ordinal,
                body_statistics.num_ordinal_leq,
                body_statistics.num_ordinal_gr,
                body_statistics.num_nominal,
                body_statistics.num_nominal_eq,
                body_statistics.num_nominal_neq,
                head_statistics.num_predictions,
                head_statistics.num_positive_predictions,
                head_statistics.num_negative_predictions,
            )

        return table
