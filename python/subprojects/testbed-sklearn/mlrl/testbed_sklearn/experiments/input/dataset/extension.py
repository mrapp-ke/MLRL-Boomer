"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow configuring the functionality to load datasets from ARFF files.
"""
from argparse import Namespace
from pathlib import Path
from typing import override

from mlrl.testbed_arff.experiments.input.sources import ArffFileSource

from mlrl.testbed.experiments.input.dataset.dataset import InputDataset
from mlrl.testbed.experiments.input.dataset.extension import DatasetFileExtension
from mlrl.testbed.experiments.input.sources import FileSource


class ArffFileExtension(DatasetFileExtension):
    """
    An extension that configures the functionality to load datasets from ARFF files.
    """

    # pylint: disable=unused-argument
    @override
    def _create_file_source(self, dataset_directory: Path, dataset: InputDataset, args: Namespace) -> FileSource:
        return ArffFileSource(dataset_directory)
