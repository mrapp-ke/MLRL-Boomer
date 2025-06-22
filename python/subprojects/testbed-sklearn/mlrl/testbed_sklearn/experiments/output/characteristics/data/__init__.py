"""
Author Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow to write characteristics of tabular datasets or predictions to different sinks.
"""
from mlrl.testbed_sklearn.experiments.output.characteristics.data.characteristics import OutputCharacteristics
from mlrl.testbed_sklearn.experiments.output.characteristics.data.characteristics_data import DataCharacteristics
from mlrl.testbed_sklearn.experiments.output.characteristics.data.writer_data import DataCharacteristicsWriter
from mlrl.testbed_sklearn.experiments.output.characteristics.data.writer_prediction import \
    PredictionCharacteristicsWriter
