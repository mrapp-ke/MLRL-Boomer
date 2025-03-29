"""
Author Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for loading data sets.
"""
from abc import ABC, abstractmethod
from typing import Optional

from mlrl.testbed.data import load_data_set, load_data_set_and_meta_data
from mlrl.testbed.dataset import Dataset
from mlrl.testbed.util.io import SUFFIX_ARFF, SUFFIX_XML, get_file_name


class DatasetLoader(ABC):
    """
    An abstract base class for all classes that allow to load datasets.
    """

    class Connection(ABC):
        """
        An abstract base class for all classes that implement a connection to a dataset.
        """

        @abstractmethod
        def load_dataset(self, feature_dtype, output_dtype, dataset_type: Optional[Dataset.Type] = None) -> Dataset:
            """
            Loads and returns a dataset.

            :param feature_dtype:   The requested type of the feature matrix
            :param output_dtype:    The requested type of the output matrix
            :param dataset_type:    The type of the dataset to be loaded or None, if the type should not be restricted
            :return:                The `Dataset` that has been loaded
            """

    @abstractmethod
    def open_connection(self) -> Connection:
        """
        Opens a connection to the dataset.

        :return: The `Connection` that has been opened
        """


class ArffDatasetLoader(DatasetLoader):
    """
    Allows to load datasets from ARFF files and corresponding Mulan XML files.
    """

    class Connection(DatasetLoader.Connection):
        """
        Stores the meta-data of an ARFF file.
        """

        def __init__(self, directory: str, dataset_name: str):
            """
            :param directory:       The path to the directory that contains the files
            :param dataset_name:    The name of the data set (without a suffix)
            """
            self.directory = directory
            self.dataset_name = dataset_name
            self.meta_data = None

        def load_dataset(self, feature_dtype, output_dtype, dataset_type: Optional[Dataset.Type] = None) -> Dataset:
            dataset_name = self.dataset_name
            file_name = dataset_type.get_file_name(dataset_name) if dataset_type else dataset_name
            arff_file_name = get_file_name(file_name, SUFFIX_ARFF)
            meta_data = self.meta_data

            if meta_data:
                x, y = load_data_set(directory=self.directory,
                                     arff_file_name=arff_file_name,
                                     meta_data=meta_data,
                                     feature_dtype=feature_dtype,
                                     output_dtype=output_dtype)
            else:
                xml_file_name = get_file_name(dataset_name, SUFFIX_XML)
                x, y, meta_data = load_data_set_and_meta_data(directory=self.directory,
                                                              arff_file_name=arff_file_name,
                                                              xml_file_name=xml_file_name,
                                                              feature_dtype=feature_dtype,
                                                              output_dtype=output_dtype)
                self.meta_data = meta_data

            return Dataset(x=x, y=y, features=meta_data.features, outputs=meta_data.outputs)

    def __init__(self, directory: str, dataset_name: str):
        """
        :param directory:       The path to the directory, where the data set is located
        :param dataset_name:    The name of the data set (without a suffix)
        """
        self.connection = ArffDatasetLoader.Connection(directory=directory, dataset_name=dataset_name)

    def open_connection(self) -> DatasetLoader.Connection:
        return self.connection
