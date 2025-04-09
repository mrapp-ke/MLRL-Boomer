"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for representing models for the calibration of probabilities via isotonic regression.
"""

from dataclasses import dataclass, field
from typing import List, Optional

from mlrl.common.config.options import Options
from mlrl.common.cython.probability_calibration import IsotonicProbabilityCalibrationModel, \
    IsotonicProbabilityCalibrationModelVisitor

from mlrl.testbed.experiments.output.data import TabularOutputData
from mlrl.testbed.experiments.output.table import ColumnWiseTable, Table
from mlrl.testbed.experiments.state import ExperimentState
from mlrl.testbed.util.format import OPTION_DECIMALS, format_number, format_table


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

        def visit_bin(self, list_index: int, threshold: float, probability: float):
            """
            See :func:`mlrl.common.cython.probability_calibration.IsotonicProbabilityCalibrationModelVisitor.visit_bin`
            """
            bin_list = self.bin_lists.setdefault(list_index, IsotonicRegressionModel.BinList())
            bin_list.thresholds.append(threshold)
            bin_list.probabilities.append(probability)

    def __init__(self,
                 calibration_model: IsotonicProbabilityCalibrationModel,
                 name: str,
                 file_name: str,
                 default_formatter_options: ExperimentState.FormatterOptions = ExperimentState.FormatterOptions(),
                 column_title_prefix: Optional[str] = None):
        """
        :param calibration_model:           The isotonic calibration model
        :param name:                        A name to be included in log messages
        :param file_name:                   A file name to be used for writing into output files
        :param default_formatter_options:   The options to be used for creating textual representations of the
                                            `ExperimentState`, the output data has been generated from
        :param column_title_prefix:         An optional prefix to be prepended to the titles of table columns that
                                            contain thresholds or probabilities
        """
        super().__init__(name=name, file_name=file_name, default_formatter_options=default_formatter_options)
        self.calibration_model = calibration_model
        self.column_title_prefix = column_title_prefix

    def _format_threshold_header(self, list_index: int) -> str:
        prefix = self.column_title_prefix
        return (prefix + ' ' if prefix else '') + str(list_index + 1) + ' thresholds'

    def _format_probability_header(self, list_index: int) -> str:
        prefix = self.column_title_prefix
        return (prefix + ' ' if prefix else '') + str(list_index + 1) + ' probabilities'

    def to_text(self, options: Options, **_) -> Optional[str]:
        """
        See :func:`mlrl.testbed.experiments.output.data.OutputData.to_text`
        """
        visitor = IsotonicRegressionModel.Visitor()
        self.calibration_model.visit(visitor)
        bin_lists = visitor.bin_lists
        decimals = options.get_int(OPTION_DECIMALS, 4)
        result = ''

        for list_index in sorted(bin_lists.keys()):
            header = [self._format_threshold_header(list_index), self._format_probability_header(list_index)]
            bin_list = bin_lists[list_index]
            rows = []

            for threshold, probability in zip(bin_list.thresholds, bin_list.probabilities):
                rows.append([
                    format_number(threshold, decimals=decimals),
                    format_number(probability, decimals=decimals),
                ])

            if result:
                result += '\n'

            result += format_table(rows, header=header)

        return result

    def to_table(self, options: Options, **_) -> Optional[Table]:
        """
        See :func:`mlrl.testbed.experiments.output.data.TabularOutputData.to_table`
        """
        visitor = IsotonicRegressionModel.Visitor()
        self.calibration_model.visit(visitor)
        bin_lists = visitor.bin_lists
        decimals = options.get_int(OPTION_DECIMALS, 0)
        table = ColumnWiseTable()

        for list_index, bin_list in bin_lists.items():
            thresholds = map(lambda value: format_number(value, decimals=decimals), bin_list.thresholds)
            table.add_column(*thresholds, header=self._format_threshold_header(list_index))
            probabilities = map(lambda value: format_number(value, decimals=decimals), bin_list.probabilities)
            table.add_column(*probabilities, header=self._format_probability_header(list_index))

        return table
