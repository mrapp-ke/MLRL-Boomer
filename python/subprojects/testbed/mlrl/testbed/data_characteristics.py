"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for printing certain characteristics of data sets. The characteristics can be written to one or several
outputs, e.g., to the console or to a file.
"""
from functools import cached_property
from typing import List, Optional

from mlrl.common.config.options import Options

from mlrl.testbed.characteristics import LABEL_CHARACTERISTICS, OUTPUT_CHARACTERISTICS, Characteristic, \
    OutputCharacteristics, density
from mlrl.testbed.dataset import AttributeType, Dataset
from mlrl.testbed.experiments.output.data import OutputData, OutputValue, TabularOutputData
from mlrl.testbed.experiments.output.writer import OutputWriter
from mlrl.testbed.experiments.problem_type import ProblemType
from mlrl.testbed.experiments.state import ExperimentState
from mlrl.testbed.format import OPTION_DECIMALS, OPTION_PERCENTAGE, format_table

OPTION_EXAMPLES = 'examples'

OPTION_FEATURES = 'features'

OPTION_NUMERICAL_FEATURES = 'numerical_features'

OPTION_ORDINAL_FEATURES = 'ordinal_features'

OPTION_NOMINAL_FEATURES = 'nominal_features'

OPTION_FEATURE_DENSITY = 'feature_density'

OPTION_FEATURE_SPARSITY = 'feature_sparsity'


class FeatureCharacteristics:
    """
    Stores characteristics of the features in a dataset.
    """

    def __init__(self, dataset: Dataset):
        """
        :param dataset: The dataset
        """
        self.dataset = dataset

    @property
    def num_examples(self) -> int:
        """
        The total number of examples.
        """
        return self.dataset.num_examples

    @property
    def num_features(self) -> int:
        """
        The total number of features.
        """
        return self.dataset.num_features

    @cached_property
    def num_nominal_features(self) -> int:
        """
        The total number of nominal features.
        """
        return self.dataset.get_num_features(AttributeType.NOMINAL)

    @cached_property
    def num_ordinal_features(self) -> int:
        """
        The total number of ordinal features.
        """
        return self.dataset.get_num_features(AttributeType.ORDINAL)

    @cached_property
    def num_numerical_features(self) -> int:
        """
        The total number of numerical features.
        """
        return self.dataset.get_num_features(AttributeType.NUMERICAL)

    @cached_property
    def feature_density(self) -> float:
        """
        The feature density.
        """
        return density(self.dataset.x)

    @property
    def feature_sparsity(self) -> float:
        """
        The feature sparsity.
        """
        return 1 - self.feature_density


FEATURE_CHARACTERISTICS: List[Characteristic] = [
    Characteristic(OPTION_EXAMPLES, 'Examples', lambda x: x.num_examples),
    Characteristic(OPTION_FEATURES, 'Features', lambda x: x.num_features),
    Characteristic(OPTION_NUMERICAL_FEATURES, 'Numerical Features', lambda x: x.num_numerical_features),
    Characteristic(OPTION_ORDINAL_FEATURES, 'Ordinal Features', lambda x: x.num_ordinal_features),
    Characteristic(OPTION_NOMINAL_FEATURES, 'Nominal Features', lambda x: x.num_nominal_features),
    Characteristic(OPTION_FEATURE_DENSITY, 'Feature Density', lambda x: x.feature_density, percentage=True),
    Characteristic(OPTION_FEATURE_SPARSITY, 'Feature Sparsity', lambda x: x.feature_sparsity, percentage=True),
]


class DataCharacteristicsWriter(OutputWriter):
    """
    Allows to write the characteristics of a data set to one or several sinks.
    """

    class DataCharacteristics(TabularOutputData):
        """
        Stores characteristics of a feature matrix and an output matrix.
        """

        def __init__(self, feature_characteristics: FeatureCharacteristics,
                     output_characteristics: OutputCharacteristics, problem_type: ProblemType):
            """
            :param feature_characteristics: The characteristics of the feature matrix
            :param output_characteristics:  The characteristics of the output matrix
            :param problem_type:            The type of the machine learning problem
            """
            super().__init__('Data characteristics', 'data_characteristics',
                             ExperimentState.FormatterOptions(include_dataset_type=False))
            self.feature_characteristics = feature_characteristics
            self.output_characteristics = output_characteristics
            classification = problem_type == ProblemType.CLASSIFICATION
            self.characteristics = LABEL_CHARACTERISTICS if classification else OUTPUT_CHARACTERISTICS

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
                    characteristic.format(self.feature_characteristics, percentage=percentage, decimals=decimals)
                ])

            for characteristic in OutputValue.filter_values(self.characteristics, options):
                rows.append([
                    characteristic.name,
                    characteristic.format(self.output_characteristics, percentage=percentage, decimals=decimals)
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
                columns[characteristic] = characteristic.format(self.feature_characteristics,
                                                                percentage=percentage,
                                                                decimals=decimals)

            for characteristic in OutputValue.filter_values(self.characteristics, options):
                columns[characteristic] = characteristic.format(self.output_characteristics,
                                                                percentage=percentage,
                                                                decimals=decimals)

            return [columns]

    def _generate_output_data(self, state: ExperimentState) -> Optional[OutputData]:
        problem_type = state.problem_type
        dataset = state.dataset
        feature_characteristics = FeatureCharacteristics(dataset)
        output_characteristics = OutputCharacteristics(problem_type, dataset.y)
        return DataCharacteristicsWriter.DataCharacteristics(feature_characteristics=feature_characteristics,
                                                             output_characteristics=output_characteristics,
                                                             problem_type=problem_type)
