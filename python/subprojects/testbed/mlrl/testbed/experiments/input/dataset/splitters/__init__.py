"""
Author Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow to split datasets into training and test datasets.
"""
from mlrl.testbed.experiments.input.dataset.splitters.splitter import DatasetSplitter
from mlrl.testbed.experiments.input.dataset.splitters.splitter_bipartition import BipartitionSplitter
from mlrl.testbed.experiments.input.dataset.splitters.splitter_cross_validation import CrossValidationSplitter
from mlrl.testbed.experiments.input.dataset.splitters.splitter_no import NoSplitter
