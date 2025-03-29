"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for implementing preprocessors.
"""
from abc import ABC, abstractmethod

from mlrl.testbed.dataset import Dataset


class Preprocessor(ABC):
    """
    An abstract base class for all classes that allow preprocessing datasets.
    """

    class Encoder(ABC):
        """
        Allows encoding datasets.
        """

        @abstractmethod
        def encode(self, dataset: Dataset) -> Dataset:
            """
            Encodes a dataset.

            :param dataset: The dataset to be encoded
            :return:        The encoded dataset
            """

    @abstractmethod
    def create_encoder(self) -> Encoder:
        """
        Creates and returns an encoder that allows preprocessing datasets.

        :return: The encoder that has been created
        """
