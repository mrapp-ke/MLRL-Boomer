"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for representing characteristics of a datasets that are part of output data.
"""
from typing import Any, List, Optional, Tuple, override

from mlrl.testbed_sklearn.experiments.dataset import TabularDataset
from mlrl.testbed_sklearn.experiments.output.characteristics.data.characteristics import Characteristic, \
    get_output_characteristics
from mlrl.testbed_sklearn.experiments.output.characteristics.data.matrix_feature import FeatureMatrix
from mlrl.testbed_sklearn.experiments.output.characteristics.data.matrix_label import LabelMatrix
from mlrl.testbed_sklearn.experiments.output.characteristics.data.matrix_output import OutputMatrix

from mlrl.testbed.experiments.context import Context
from mlrl.testbed.experiments.data import TabularProperties
from mlrl.testbed.experiments.output.data import OutputValue, TabularOutputData
from mlrl.testbed.experiments.problem_domain import ClassificationProblem, ProblemDomain
from mlrl.testbed.experiments.table import RowWiseTable, Table
from mlrl.testbed.util.format import OPTION_DECIMALS, OPTION_PERCENTAGE

from mlrl.util.options import Options


class DataCharacteristics(TabularOutputData):
    """
    Represents characteristics of a tabular dataset that are part of output data.
    """

    PROPERTIES = TabularProperties(file_name='data_characteristics', name='Data characteristics')

    CONTEXT = Context(include_dataset_type=False)

    OPTION_EXAMPLES = 'examples'

    OPTION_FEATURES = 'features'

    OPTION_NUMERICAL_FEATURES = 'numerical_features'

    OPTION_ORDINAL_FEATURES = 'ordinal_features'

    OPTION_NOMINAL_FEATURES = 'nominal_features'

    OPTION_FEATURE_DENSITY = 'feature_density'

    OPTION_FEATURE_SPARSITY = 'feature_sparsity'

    def __init__(self, values: List[Tuple[Characteristic, Any]]):
        """
        :param problem_domain:  The problem domain, the dataset is concerned with
        :param dataset:         The dataset
        """
        super().__init__(properties=self.PROPERTIES, context=self.CONTEXT)
        self.values = values
        self.characteristics = [characteristic for characteristic, _ in values]

    @staticmethod
    def from_dataset(problem_domain: ProblemDomain, dataset: TabularDataset) -> 'DataCharacteristics':
        """
        Creates and returns `DataCharacteristics` from a given dataset.

        :param problem_domain:  The problem domain, the dataset is concerned with
        :param dataset:         The dataset
        :return:                The `DataCharacteristics` that have been created
        """
        feature_matrix = FeatureMatrix(dataset=dataset)
        output_characteristics = get_output_characteristics(problem_domain)

        if isinstance(problem_domain, ClassificationProblem):
            output_matrix: OutputMatrix = LabelMatrix(values=dataset.y)
        else:
            output_matrix = OutputMatrix(values=dataset.y)

        return DataCharacteristics([(characteristic, characteristic.function(feature_matrix))
                                    for characteristic in FEATURE_CHARACTERISTICS]
                                   + [(characteristic, characteristic.function(output_matrix))
                                      for characteristic in output_characteristics])

    @override
    def to_text(self, options: Options, **kwargs) -> Optional[str]:
        """
        See :func:`mlrl.testbed.experiments.output.data.TextualOutputData.to_text`
        """
        kwargs = dict(kwargs) | {OPTION_DECIMALS: 2}
        table = self.to_table(options, **kwargs)
        return table.format() if table else None

    @override
    def to_table(self, options: Options, **kwargs) -> Optional[Table]:
        """
        See :func:`mlrl.testbed.experiments.output.data.TabularOutputData.to_table`
        """
        percentage = options.get_bool(OPTION_PERCENTAGE, kwargs.get(OPTION_PERCENTAGE, True))
        decimals = options.get_int(OPTION_DECIMALS, kwargs.get(OPTION_DECIMALS, 0))
        characteristics = OutputValue.filter_values(self.characteristics, options)
        values = [
            characteristic.format(value, percentage=percentage, decimals=decimals)
            for characteristic, value in self.values
        ]
        return RowWiseTable(*characteristics).add_row(*values)


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
