"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for representing models for the calibration of probabilities via isotonic regression.
"""

from typing import List, Optional, Tuple

from mlrl.common.config.options import Options
from mlrl.common.cython.probability_calibration import IsotonicProbabilityCalibrationModel, \
    IsotonicProbabilityCalibrationModelVisitor

from mlrl.testbed.experiments.output.data import TabularOutputData
from mlrl.testbed.experiments.state import ExperimentState
from mlrl.testbed.util.format import OPTION_DECIMALS, format_number, format_table


class IsotonicRegressionModel(TabularOutputData):
    """
    Represents an isotonic regression model.
    """

    Bin = List[Tuple[float, float]]

    class Visitor(IsotonicProbabilityCalibrationModelVisitor):
        """
        Accesses the thresholds and probabilities stored by an `IsotonicProbabilityCalibrationModel` and stores them in
        bins.
        """

        def __init__(self):
            self.bins = {}

        def visit_bin(self, list_index: int, threshold: float, probability: float):
            """
            See :func:`mlrl.common.cython.probability_calibration.IsotonicProbabilityCalibrationModelVisitor.visit_bin`
            """
            bin_list = self.bins.setdefault(list_index, [])
            bin_list.append((threshold, probability))

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

    def _format_threshold_column(self, list_index: int) -> str:
        prefix = self.column_title_prefix
        return (prefix + ' ' if prefix else '') + str(list_index + 1) + ' thresholds'

    def _format_probability_column(self, list_index: int) -> str:
        prefix = self.column_title_prefix
        return (prefix + ' ' if prefix else '') + str(list_index + 1) + ' probabilities'

    def to_text(self, options: Options, **_) -> Optional[str]:
        """
        See :func:`mlrl.testbed.experiments.output.data.OutputData.to_text`
        """
        visitor = IsotonicRegressionModel.Visitor()
        self.calibration_model.visit(visitor)
        bins = visitor.bins
        decimals = options.get_int(OPTION_DECIMALS, 4)
        result = ''

        for list_index in sorted(bins.keys()):
            header = [self._format_threshold_column(list_index), self._format_probability_column(list_index)]
            rows = []

            for threshold, probability in bins[list_index]:
                rows.append([
                    format_number(threshold, decimals=decimals),
                    format_number(probability, decimals=decimals),
                ])

            if result:
                result += '\n'

            result += format_table(rows, header=header)

        return result

    def to_table(self, options: Options, **_) -> Optional[TabularOutputData.Table]:
        """
        See :func:`mlrl.testbed.experiments.output.data.TabularOutputData.to_table`
        """
        visitor = IsotonicRegressionModel.Visitor()
        self.calibration_model.visit(visitor)
        bins = visitor.bins
        decimals = options.get_int(OPTION_DECIMALS, 0)
        rows = []
        end = False
        i = 0

        while not end:
            columns = {}
            end = True

            for list_index, bin_list in bins.items():
                column_probability = self._format_probability_column(list_index)
                column_threshold = self._format_threshold_column(list_index)

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
