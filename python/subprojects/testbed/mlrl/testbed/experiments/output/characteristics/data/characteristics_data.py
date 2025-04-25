"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for representing characteristics of a datasets that are part of output data.
"""
from itertools import chain
from typing import Optional

from mlrl.common.config.options import Options

from mlrl.testbed.experiments.data import Data
from mlrl.testbed.experiments.dataset import Dataset
from mlrl.testbed.experiments.output.characteristics.data.characteristics import LABEL_CHARACTERISTICS, \
    OUTPUT_CHARACTERISTICS, Characteristic
from mlrl.testbed.experiments.output.characteristics.data.matrix_feature import FeatureMatrix
from mlrl.testbed.experiments.output.characteristics.data.matrix_label import LabelMatrix
from mlrl.testbed.experiments.output.characteristics.data.matrix_output import OutputMatrix
from mlrl.testbed.experiments.output.data import OutputData, OutputValue, TabularOutputData
from mlrl.testbed.experiments.problem_type import ProblemType
from mlrl.testbed.experiments.table import RowWiseTable, Table
from mlrl.testbed.util.format import OPTION_DECIMALS, OPTION_PERCENTAGE


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
        super().__init__(OutputData.Properties(file_name='data_characteristics', name='Data characteristics'),
                         Data.Context(include_dataset_type=False))
        self.feature_matrix = FeatureMatrix(dataset=dataset)

        if problem_type == ProblemType.CLASSIFICATION:
            self.output_characteristics = LABEL_CHARACTERISTICS
            self.output_matrix = LabelMatrix(values=dataset.y)
        else:
            self.output_characteristics = OUTPUT_CHARACTERISTICS
            self.output_matrix = OutputMatrix(values=dataset.y)

    def to_text(self, options: Options, **kwargs) -> Optional[str]:
        """
        See :func:`mlrl.testbed.experiments.output.data.TextualOutputData.to_text`
        """
        kwargs = dict(kwargs) | {OPTION_DECIMALS: 2}
        return self.to_table(options, **kwargs).format()

    def to_table(self, options: Options, **kwargs) -> Optional[Table]:
        """
        See :func:`mlrl.testbed.experiments.output.data.TabularOutputData.to_table`
        """
        percentage = options.get_bool(OPTION_PERCENTAGE, kwargs.get(OPTION_PERCENTAGE, True))
        decimals = options.get_int(OPTION_DECIMALS, kwargs.get(OPTION_DECIMALS, 0))
        feature_characteristics = OutputValue.filter_values(FEATURE_CHARACTERISTICS, options)
        output_characteristics = OutputValue.filter_values(self.output_characteristics, options)
        values = chain(
            map(
                lambda characteristic: characteristic.format(
                    self.feature_matrix, percentage=percentage, decimals=decimals), feature_characteristics),
            map(
                lambda characteristic: characteristic.format(
                    self.output_matrix, percentage=percentage, decimals=decimals), output_characteristics),
        )
        return RowWiseTable(*chain(feature_characteristics, output_characteristics)).add_row(*values)


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
