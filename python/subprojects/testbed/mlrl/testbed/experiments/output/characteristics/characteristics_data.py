"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for representing characteristics of a datasets that are part of output data.
"""
from typing import Optional

from mlrl.common.config.options import Options

from mlrl.testbed.dataset import Dataset
from mlrl.testbed.experiments.output.characteristics.characteristics_output import LABEL_CHARACTERISTICS, \
    OUTPUT_CHARACTERISTICS, Characteristic
from mlrl.testbed.experiments.output.characteristics.matrix_feature import FeatureMatrix
from mlrl.testbed.experiments.output.characteristics.matrix_label import LabelMatrix
from mlrl.testbed.experiments.output.characteristics.matrix_output import OutputMatrix
from mlrl.testbed.experiments.output.data import OutputValue, TabularOutputData
from mlrl.testbed.experiments.problem_type import ProblemType
from mlrl.testbed.experiments.state import ExperimentState
from mlrl.testbed.util.format import OPTION_DECIMALS, OPTION_PERCENTAGE, format_table


class DataCharacteristics(TabularOutputData):
    """
    Represents characteristics of a dataset that are part of output data.
    """

    OPTION_EXAMPLES = 'examples'

    OPTION_FEATURES = 'features'

    OPTION_NUMERICAL_FEATURES = 'numerical_features'

    OPTION_ORDINAL_FEATURES = 'ordinal_features'

    OPTION_NOMINAL_FEATURES = 'nominal_features'

    OPTION_FEATURE_DENSITY = 'feature_density'

    OPTION_FEATURE_SPARSITY = 'feature_sparsity'

    def __init__(self, problem_type: ProblemType, dataset: Dataset):
        """
        :param problem_type:    The type of the machine learning problem, the dataset is concerned with
        :param dataset:         The dataset
        """
        super().__init__(name='Data characteristics',
                         file_name='data_characteristics',
                         default_formatter_options=ExperimentState.FormatterOptions(include_dataset_type=False))
        self.feature_matrix = FeatureMatrix(dataset=dataset)

        if problem_type == ProblemType.CLASSIFICATION:
            self.output_characteristics = LABEL_CHARACTERISTICS
            self.output_matrix = LabelMatrix(values=dataset.y)
        else:
            self.output_characteristics = OUTPUT_CHARACTERISTICS
            self.output_matrix = OutputMatrix(values=dataset.y)

    def to_text(self, options: Options, **_) -> Optional[str]:
        """
        See :func:`mlrl.testbed.experiments.output.data.OutputData.to_text`
        """
        percentage = options.get_bool(OPTION_PERCENTAGE, True)
        decimals = options.get_int(OPTION_DECIMALS, 2)
        rows = []

        for characteristic in OutputValue.filter_values(FEATURE_CHARACTERISTICS, options):
            rows.append([
                characteristic.name,
                characteristic.format(self.feature_matrix, percentage=percentage, decimals=decimals)
            ])

        for characteristic in OutputValue.filter_values(self.output_characteristics, options):
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

        for characteristic in OutputValue.filter_values(FEATURE_CHARACTERISTICS, options):
            columns[characteristic] = characteristic.format(self.feature_matrix,
                                                            percentage=percentage,
                                                            decimals=decimals)

        for characteristic in OutputValue.filter_values(self.output_characteristics, options):
            columns[characteristic] = characteristic.format(self.output_matrix,
                                                            percentage=percentage,
                                                            decimals=decimals)

        return [columns]


FEATURE_CHARACTERISTICS = [
    Characteristic(
        option_key=DataCharacteristics.OPTION_EXAMPLES,
        name='Examples',
        function=lambda x: x.num_examples,
    ),
    Characteristic(
        option_key=DataCharacteristics.OPTION_FEATURES,
        name='Features',
        function=lambda x: x.num_features,
    ),
    Characteristic(
        option_key=DataCharacteristics.OPTION_NUMERICAL_FEATURES,
        name='Numerical Features',
        function=lambda x: x.num_numerical_features,
    ),
    Characteristic(
        option_key=DataCharacteristics.OPTION_ORDINAL_FEATURES,
        name='Ordinal Features',
        function=lambda x: x.num_ordinal_features,
    ),
    Characteristic(
        option_key=DataCharacteristics.OPTION_NOMINAL_FEATURES,
        name='Nominal Features',
        function=lambda x: x.num_nominal_features,
    ),
    Characteristic(
        option_key=DataCharacteristics.OPTION_FEATURE_DENSITY,
        name='Feature Density',
        function=lambda x: x.feature_density,
        percentage=True,
    ),
    Characteristic(
        option_key=DataCharacteristics.OPTION_FEATURE_SPARSITY,
        name='Feature Sparsity',
        function=lambda x: x.feature_sparsity,
        percentage=True,
    ),
]
