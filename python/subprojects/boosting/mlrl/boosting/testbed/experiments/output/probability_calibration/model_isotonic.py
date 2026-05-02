"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for representing models for the calibration of probabilities via isotonic regression.
"""

from dataclasses import dataclass, field
from typing import override
from rich.console import ConsoleRenderable, Group
from rich.text import Text
from mlrl.common.cython.probability_calibration import (
    IsotonicProbabilityCalibrationModel,
    IsotonicProbabilityCalibrationModelVisitor,
)

from mlrl.testbed.experiments.context import Context
from mlrl.testbed.experiments.data import TabularProperties
from mlrl.testbed.experiments.output.data import TabularOutputData
from mlrl.testbed.experiments.table import ColumnWiseTable, Table
from mlrl.testbed.util.format import OPTION_DECIMALS

from mlrl.util.format import format_value
from mlrl.util.options import Options


class IsotonicRegressionModel(TabularOutputData):
    """
    Represents an isotonic regression model.
    """

    COLUMN_THRESHOLDS = 'thresholds'

    COLUMN_PROBABILITIES = 'probabilities'

    @dataclass
    class BinList:
        """
        A list of bins that is contained in an isotonic regression model.

        Attributes:
            thresholds:     A list the contains the thresholds of individual bins
            probabilities:  A list that contains the probabilities of individual bins
        """

        thresholds: list[float] = field(default_factory=list)
        probabilities: list[float] = field(default_factory=list)

    class Visitor(IsotonicProbabilityCalibrationModelVisitor):
        """
        Accesses the thresholds and probabilities stored by an `IsotonicProbabilityCalibrationModel` and stores them in
        bins.
        """

        def __init__(self):
            self.bin_lists: dict[int, IsotonicRegressionModel.BinList] = {}

        @override
        def visit_bin(self, list_index: int, threshold: float, probability: float):
            """
            See :func:`mlrl.common.cython.probability_calibration.IsotonicProbabilityCalibrationModelVisitor.visit_bin`
            """
            bin_list = self.bin_lists.setdefault(list_index, IsotonicRegressionModel.BinList())
            bin_list.thresholds.append(threshold)
            bin_list.probabilities.append(probability)

    def __init__(
        self,
        bin_lists: dict[int, 'IsotonicRegressionModel.BinList'],
        properties: TabularProperties,
        context: Context = Context(),
        column_title_prefix: str | None = None,
    ):
        """
        :param bin_lists:           A dictionary that stores lists of bins contained in an isotonic regression model,
                                    mapped to indices
        :param properties:          The properties of the output data
        :param context:             A `Context` to be used by default for finding a suitable sink this output data can
                                    be written to
        :param column_title_prefix: An optional prefix to be prepended to the titles of table columns that contain
                                    thresholds or probabilities
        """
        super().__init__(properties=properties, context=context)
        self.bin_lists = bin_lists
        self.column_title_prefix = column_title_prefix

    @staticmethod
    def from_calibration_model(
        calibration_model: IsotonicProbabilityCalibrationModel,
        properties: TabularProperties,
        context: Context = Context(),
        column_title_prefix: str | None = None,
    ) -> 'IsotonicRegressionModel':
        """
        Creates and returns an `IsotonicRegressionModel` from a given `IsotonicProbabilityCalibrationModel`.

        :param calibration_model:   An `IsotonicProbabilityCalibrationModel`
        :param properties:          The properties of the output data
        :param context:             A `Context` to be used by default for finding a suitable sink this output data can
                                    be written to
        :param column_title_prefix: An optional prefix to be prepended to the titles of table columns that contain
                                    thresholds or probabilities
        :return:                    The `IsotonicRegressionModel` that has been created
        """
        visitor = IsotonicRegressionModel.Visitor()
        calibration_model.visit(visitor)
        bin_lists = visitor.bin_lists
        return IsotonicRegressionModel(
            bin_lists=bin_lists, properties=properties, context=context, column_title_prefix=column_title_prefix
        )

    def _format_threshold_header(self, list_index: int) -> str:
        prefix = self.column_title_prefix
        return f'{(f"{prefix} " if prefix else "")}{list_index + 1} {self.COLUMN_THRESHOLDS}'

    def _format_probability_header(self, list_index: int) -> str:
        prefix = self.column_title_prefix
        return f'{(f"{prefix} " if prefix else "")}{list_index + 1} {self.COLUMN_PROBABILITIES}'

    @override
    def to_text(self, options: Options, **kwargs) -> str | ConsoleRenderable | None:
        """
        See :func:`mlrl.testbed.experiments.output.data.TextualOutputData.to_text`
        """
        kwargs = dict(kwargs) | {OPTION_DECIMALS: 4}
        table = self.to_table(options, **kwargs)

        if table:
            columns = table.columns
            renderables: list[ConsoleRenderable] = []

            for list_index, _ in enumerate(range(0, table.num_columns, 2)):
                bin_list_table = ColumnWiseTable()
                bin_list_table.add_column(
                    *filter(lambda value: value is not None, next(columns)),
                    header=self._format_threshold_header(list_index),
                )
                bin_list_table.add_column(
                    *filter(lambda value: value is not None, next(columns)),
                    header=self._format_probability_header(list_index),
                )

                if renderables:
                    renderables.append(Text(''))

                renderables.append(
                    bin_list_table.to_rich_table(
                        auto_rotate=False,
                        table_format=Table.Format.SIMPLE,
                        column_styles=[Table.COLUMN_STYLE_VALUE, Table.COLUMN_STYLE_VALUE],
                    )
                )

            return Group(*renderables) if renderables else None

        return None

    @override
    def to_table(self, options: Options, **kwargs) -> Table | None:
        """
        See :func:`mlrl.testbed.experiments.output.data.TabularOutputData.to_table`
        """
        decimals = options.get_int(OPTION_DECIMALS, kwargs.get(OPTION_DECIMALS, 0))
        table = ColumnWiseTable()

        for list_index, bin_list in self.bin_lists.items():
            thresholds = map(lambda value: format_value(value, decimals=decimals), bin_list.thresholds)
            table.add_column(*thresholds, header=self._format_threshold_header(list_index))
            probabilities = map(lambda value: format_value(value, decimals=decimals), bin_list.probabilities)
            table.add_column(*probabilities, header=self._format_probability_header(list_index))

        return table
