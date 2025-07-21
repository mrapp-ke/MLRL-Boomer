"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for representing models for the calibration of probabilities via isotonic regression.
"""

from dataclasses import dataclass, field
from typing import List, Optional, override

from mlrl.common.cython.probability_calibration import IsotonicProbabilityCalibrationModel, \
    IsotonicProbabilityCalibrationModelVisitor

from mlrl.testbed.experiments.context import Context
from mlrl.testbed.experiments.output.data import OutputData, TabularOutputData
from mlrl.testbed.experiments.table import ColumnWiseTable, Table
from mlrl.testbed.util.format import OPTION_DECIMALS, format_number

from mlrl.util.options import Options


class IsotonicRegressionModel(TabularOutputData):
    """
    Represents an isotonic regression model.
    """

    @dataclass
    class BinList:
        """
        A list of bins that is contained in an isotonic regression model.

        Attributes:
            thresholds:     A list the contains the thresholds of individual bins
            probabilities:  A list that contains the probabilities of individual bins
        """
        thresholds: List[float] = field(default_factory=list)
        probabilities: List[float] = field(default_factory=list)

    class Visitor(IsotonicProbabilityCalibrationModelVisitor):
        """
        Accesses the thresholds and probabilities stored by an `IsotonicProbabilityCalibrationModel` and stores them in
        bins.
        """

        def __init__(self):
            self.bin_lists = {}

        @override
        def visit_bin(self, list_index: int, threshold: float, probability: float):
            """
            See :func:`mlrl.common.cython.probability_calibration.IsotonicProbabilityCalibrationModelVisitor.visit_bin`
            """
            bin_list = self.bin_lists.setdefault(list_index, IsotonicRegressionModel.BinList())
            bin_list.thresholds.append(threshold)
            bin_list.probabilities.append(probability)

    def __init__(self,
                 calibration_model: IsotonicProbabilityCalibrationModel,
                 properties: OutputData.Properties,
                 context: Context = Context(),
                 column_title_prefix: Optional[str] = None):
        """
        :param calibration_model:   The isotonic calibration model
        :param properties:          The properties of the output data
        :param context:             A `Context` to be used by default for finding a suitable sink this output data can
                                    be written to
        :param column_title_prefix: An optional prefix to be prepended to the titles of table columns that contain
                                    thresholds or probabilities
        """
        super().__init__(properties=properties, context=context)
        self.calibration_model = calibration_model
        self.column_title_prefix = column_title_prefix

    def _format_threshold_header(self, list_index: int) -> str:
        prefix = self.column_title_prefix
        return (prefix + ' ' if prefix else '') + str(list_index + 1) + ' thresholds'

    def _format_probability_header(self, list_index: int) -> str:
        prefix = self.column_title_prefix
        return (prefix + ' ' if prefix else '') + str(list_index + 1) + ' probabilities'

    @override
    def to_text(self, options: Options, **kwargs) -> Optional[str]:
        """
        See :func:`mlrl.testbed.experiments.output.data.TextualOutputData.to_text`
        """
        kwargs = dict(kwargs) | {OPTION_DECIMALS: 4}
        table = self.to_table(options, **kwargs)

        if table:
            columns = table.columns
            result = ''

            for list_index, _ in enumerate(range(0, table.num_columns, 2)):
                bin_list_table = ColumnWiseTable()
                bin_list_table.add_column(*filter(lambda value: value is not None, next(columns)),
                                          header=self._format_threshold_header(list_index))
                bin_list_table.add_column(*filter(lambda value: value is not None, next(columns)),
                                          header=self._format_probability_header(list_index))

                if result:
                    result += '\n'

                result += bin_list_table.format(auto_rotate=False)

            return result

        return None

    @override
    def to_table(self, options: Options, **kwargs) -> Optional[Table]:
        """
        See :func:`mlrl.testbed.experiments.output.data.TabularOutputData.to_table`
        """
        visitor = IsotonicRegressionModel.Visitor()
        self.calibration_model.visit(visitor)
        bin_lists = visitor.bin_lists
        decimals = options.get_int(OPTION_DECIMALS, kwargs.get(OPTION_DECIMALS, 0))
        table = ColumnWiseTable()

        for list_index, bin_list in bin_lists.items():
            thresholds = map(lambda value: format_number(value, decimals=decimals), bin_list.thresholds)
            table.add_column(*thresholds, header=self._format_threshold_header(list_index))
            probabilities = map(lambda value: format_number(value, decimals=decimals), bin_list.probabilities)
            table.add_column(*probabilities, header=self._format_probability_header(list_index))

        return table
