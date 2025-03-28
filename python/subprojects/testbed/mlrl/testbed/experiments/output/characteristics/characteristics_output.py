"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for representing characteristics of an output matrix that are part of output data.
"""
from typing import Optional

from mlrl.common.config.options import Options

from mlrl.testbed.experiments.output.characteristics.characteristic import Characteristic
from mlrl.testbed.experiments.output.characteristics.matrix_output import OutputMatrix
from mlrl.testbed.experiments.output.data import OutputValue, TabularOutputData
from mlrl.testbed.experiments.problem_type import ProblemType
from mlrl.testbed.experiments.state import ExperimentState
from mlrl.testbed.util.format import OPTION_DECIMALS, OPTION_PERCENTAGE, format_table

OPTION_OUTPUTS = 'outputs'

OPTION_OUTPUT_DENSITY = 'output_density'

OPTION_OUTPUT_SPARSITY = 'output_sparsity'

OPTION_LABEL_IMBALANCE_RATIO = 'label_imbalance_ratio'

OPTION_LABEL_CARDINALITY = 'label_cardinality'

OPTION_DISTINCT_LABEL_VECTORS = 'distinct_label_vectors'

OUTPUT_CHARACTERISTICS = [
    Characteristic(OPTION_OUTPUTS, 'Outputs', lambda x: x.num_outputs),
    Characteristic(OPTION_OUTPUT_DENSITY, 'Output Density', lambda x: x.output_density, percentage=True),
    Characteristic(OPTION_OUTPUT_SPARSITY, 'Output Sparsity', lambda x: x.output_sparsity, percentage=True)
]

LABEL_CHARACTERISTICS = OUTPUT_CHARACTERISTICS + [
    Characteristic(OPTION_LABEL_IMBALANCE_RATIO, 'Label Imbalance Ratio', lambda x: x.avg_label_imbalance_ratio),
    Characteristic(OPTION_LABEL_CARDINALITY, 'Label Cardinality', lambda x: x.avg_label_cardinality),
    Characteristic(OPTION_DISTINCT_LABEL_VECTORS, 'Distinct Label Vectors', lambda x: x.num_distinct_label_vectors)
]


class OutputCharacteristics(TabularOutputData):
    """
    Represents characteristics of an output matrix that are part of output data.
    """

    def __init__(self,
                 problem_type: ProblemType,
                 output_matrix: OutputMatrix,
                 name: str,
                 file_name: str,
                 default_formatter_options: ExperimentState.FormatterOptions = ExperimentState.FormatterOptions()):
        """

        :param problem_type:                The type of the machine learning problem, the output matrix corresponds to
        :param output_matrix:               An output matrix
        :param name:                        A name to be included in log messages
        :param file_name:                   A file name to be used for writing into output files
        :param default_formatter_options:   The options to be used for creating textual representations of the
                                            `ExperimentState`, the output data has been generated from
        """
        super().__init__(name=name, file_name=file_name, default_formatter_options=default_formatter_options)
        self.output_matrix = output_matrix

        if problem_type == ProblemType.CLASSIFICATION:
            self.characteristics = LABEL_CHARACTERISTICS
        else:
            self.characteristics = OUTPUT_CHARACTERISTICS

    def to_text(self, options: Options, **_) -> Optional[str]:
        """
        See :func:`mlrl.testbed.experiments.output.data.OutputData.to_text`
        """
        percentage = options.get_bool(OPTION_PERCENTAGE, True)
        decimals = options.get_int(OPTION_DECIMALS, 2)
        rows = []

        for characteristic in OutputValue.filter_values(self.characteristics, options):
            rows.append([
                characteristic.name,
                characteristic.format(self.output_matrix, percentage=percentage, decimals=decimals)
            ])

        return format_table(rows)

    def to_table(self, options: Options, **_) -> Optional[TabularOutputData.Table]:
        """
        See :func:`mlrl.testbed.experiments.output.data.TabularOutputData.to_table`
        """
        percentage = options.get_bool(OPTION_PERCENTAGE, True)
        decimals = options.get_int(OPTION_DECIMALS, 0)
        columns = {}

        for characteristic in Characteristic.filter_values(self.characteristics, options):
            columns[characteristic] = characteristic.format(self.output_matrix,
                                                            percentage=percentage,
                                                            decimals=decimals)

        return [columns]
