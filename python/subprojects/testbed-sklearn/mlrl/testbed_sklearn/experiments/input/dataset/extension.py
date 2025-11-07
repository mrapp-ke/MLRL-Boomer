"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow configuring the functionality to load datasets from ARFF files.
"""
from argparse import Namespace
from pathlib import Path
from typing import Sequence, override

from mlrl.testbed_arff.experiments.input.sources import ArffFileSource

from mlrl.testbed_sklearn.experiments.input.sources.source_svm import SvmFileSource

from mlrl.testbed.experiments.input.dataset.dataset import InputDataset
from mlrl.testbed.experiments.input.dataset.extension import DatasetFileExtension
from mlrl.testbed.experiments.input.sources import FileSource


class ArffFileExtension(DatasetFileExtension):
    """
    An extension that configures the functionality to load datasets from ARFF files.
    """

    def __init__(self):
        super().__init__(file_type=ArffFileSource.SUFFIX_ARFF)

    # pylint: disable=unused-argument
    @override
    def _create_file_sources(self, dataset_directory: Path, dataset: InputDataset,
                             args: Namespace) -> Sequence[FileSource]:
        return [ArffFileSource(dataset_directory)]


class SvmFileExtension(DatasetFileExtension):
    """
    An extension that configures the functionality to load datasets from SVM files.
    """

    def __init__(self):
        super().__init__(file_type=SvmFileSource.SUFFIX_SVM)

    # pylint: disable=unused-argument
    @override
    def _create_file_sources(self, dataset_directory: Path, dataset: InputDataset,
                             args: Namespace) -> Sequence[FileSource]:
        return [SvmFileSource(dataset_directory)]
