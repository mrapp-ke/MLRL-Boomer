"""
Author Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow to write output data to different sinks.
"""
from mlrl.testbed.experiments.output.sinks.sink import FileSink, Sink
from mlrl.testbed.experiments.output.sinks.sink_csv import CsvFileSink
from mlrl.testbed.experiments.output.sinks.sink_log import LogSink
from mlrl.testbed.experiments.output.sinks.sink_pickle import PickleFileSink
from mlrl.testbed.experiments.output.sinks.sink_text import TextFileSink
from mlrl.testbed.experiments.output.sinks.sink_yaml import YamlFileSink
