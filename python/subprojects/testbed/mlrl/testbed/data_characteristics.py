"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for printing certain characteristics of data sets. The characteristics can be written to one or several
outputs, e.g., to the console or to a file.
"""
from functools import cached_property
from typing import Any, Dict, List, Optional

from mlrl.common.options import Options

from mlrl.testbed.characteristics import LABEL_CHARACTERISTICS, OUTPUT_CHARACTERISTICS, Characteristic, \
    OutputCharacteristics, density
from mlrl.testbed.data import FeatureType, MetaData
from mlrl.testbed.data_splitting import DataSplit, DataType
from mlrl.testbed.format import OPTION_DECIMALS, OPTION_PERCENTAGE, filter_formatters, format_table
from mlrl.testbed.output_writer import Formattable, OutputWriter, Tabularizable
from mlrl.testbed.prediction_scope import PredictionScope, PredictionType
from mlrl.testbed.problem_type import ProblemType

OPTION_EXAMPLES = 'examples'

OPTION_FEATURES = 'features'

OPTION_NUMERICAL_FEATURES = 'numerical_features'

OPTION_ORDINAL_FEATURES = 'ordinal_features'

OPTION_NOMINAL_FEATURES = 'nominal_features'

OPTION_FEATURE_DENSITY = 'feature_density'

OPTION_FEATURE_SPARSITY = 'feature_sparsity'


class FeatureCharacteristics:
    """
    Stores characteristics of a feature matrix.
    """

    def __init__(self, meta_data: MetaData, x):
        """
        :param meta_data:   The meta-data of the data set
        :param x:           A `numpy.ndarray`, `scipy.sparse.spmatrix` or `scipy.sparse.sparray`, shape
                            `(num_examples, num_features)`, that stores the feature values
        """
        self._x = x
        self._meta_data = meta_data
        self.num_examples = x.shape[0]

    @property
    def num_features(self):
        """
        The total number of features.
        """
        return self._meta_data.get_num_features()

    @cached_property
    def num_nominal_features(self):
        """
        The total number of nominal features.
        """
        return self._meta_data.get_num_features({FeatureType.NOMINAL})

    @cached_property
    def num_ordinal_features(self):
        """
        The total number of ordinal features.
        """
        return self._meta_data.get_num_features({FeatureType.ORDINAL})

    @cached_property
    def num_numerical_features(self):
        """
        The total number of numerical features.
        """
        return self._meta_data.get_num_features({FeatureType.NUMERICAL})

    @cached_property
    def feature_density(self):
        """
        The feature density.
        """
        return density(self._x)

    @property
    def feature_sparsity(self):
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

    class DataCharacteristics(Formattable, Tabularizable):
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
            self.feature_characteristics = feature_characteristics
            self.output_characteristics = output_characteristics
            classification = problem_type == ProblemType.CLASSIFICATION
            self.output_formatters = LABEL_CHARACTERISTICS if classification else OUTPUT_CHARACTERISTICS

        def format(self, options: Options, **_) -> str:
            """
            See :func:`mlrl.testbed.output_writer.Formattable.format`
            """
            percentage = options.get_bool(OPTION_PERCENTAGE, True)
            decimals = options.get_int(OPTION_DECIMALS, 2)
            rows = []

            for formatter in filter_formatters(FEATURE_CHARACTERISTICS, [options]):
                rows.append([
                    formatter.name,
                    formatter.format(self.feature_characteristics, percentage=percentage, decimals=decimals)
                ])

            for formatter in filter_formatters(self.output_formatters, [options]):
                rows.append([
                    formatter.name,
                    formatter.format(self.output_characteristics, percentage=percentage, decimals=decimals)
                ])

            return format_table(rows)

        def tabularize(self, options: Options, **_) -> Optional[List[Dict[str, str]]]:
            """
            See :func:`mlrl.testbed.output_writer.Tabularizable.tabularize`
            """
            percentage = options.get_bool(OPTION_PERCENTAGE, True)
            decimals = options.get_int(OPTION_DECIMALS, 0)
            columns = {}

            for formatter in filter_formatters(FEATURE_CHARACTERISTICS, [options]):
                columns[formatter] = formatter.format(self.feature_characteristics,
                                                      percentage=percentage,
                                                      decimals=decimals)

            for formatter in filter_formatters(self.output_formatters, [options]):
                columns[formatter] = formatter.format(self.output_characteristics,
                                                      percentage=percentage,
                                                      decimals=decimals)

            return [columns]

    class LogSink(OutputWriter.LogSink):
        """
        Allows to write the characteristics of a data set to the console.
        """

        def __init__(self, options: Options = Options()):
            super().__init__(title='Data characteristics', options=options)

    class CsvSink(OutputWriter.CsvSink):
        """
        Allows to write the characteristics of a data set to a CSV file.
        """

        def __init__(self, output_dir: str, options: Options = Options()):
            super().__init__(output_dir=output_dir, file_name='data_characteristics', options=options)

    # pylint: disable=unused-argument
    def _generate_output_data(self, problem_type: ProblemType, meta_data: MetaData, x, y, data_split: DataSplit,
                              learner, data_type: Optional[DataType], prediction_type: Optional[PredictionType],
                              prediction_scope: Optional[PredictionScope], predictions: Optional[Any],
                              train_time: float, predict_time: float) -> Optional[Any]:
        feature_characteristics = FeatureCharacteristics(meta_data, x)
        output_characteristics = OutputCharacteristics(y, problem_type)
        return DataCharacteristicsWriter.DataCharacteristics(feature_characteristics=feature_characteristics,
                                                             output_characteristics=output_characteristics,
                                                             problem_type=problem_type)
