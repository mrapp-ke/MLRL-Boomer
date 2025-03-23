"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for preprocessing datasets.
"""
import logging as log

from abc import ABC, abstractmethod
from dataclasses import replace

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder as SkLearnOneHotEncoder

from mlrl.common.data.arrays import is_sparse

from mlrl.testbed.data import AttributeType
from mlrl.testbed.dataset import Dataset


class Preprocessor(ABC):
    """
    An abstract base class for all classes that allow to preprocess datasets.
    """

    class Encoder(ABC):
        """
        Allows to encode datasets.
        """

        @abstractmethod
        def encode(self, dataset: Dataset) -> Dataset:
            """
            Encodes a `Dataset`.

            :param dataset: The `Dataset` to be encoded
            :return:        The encoded `Dataset`
            """

    @abstractmethod
    def create_encoder(self) -> Encoder:
        """
        Creates and returns an `Encoder` that allows preprocessing datasets.

        :return: The `Encoder` that has been created
        """


class OneHotEncoder(Preprocessor):
    """
    Allows to one-hot encode all nominal features contained in a data set, if any.
    """

    class Encoder(Preprocessor.Encoder):
        """
        Allows to one-hot encode all nominal features contained in a data set.
        """

        def __init__(self):
            self.encoder = None

        def encode(self, dataset: Dataset) -> Dataset:
            nominal_indices = dataset.get_feature_indices(AttributeType.NOMINAL)
            num_nominal_features = len(nominal_indices)
            log.info('Data set contains %s nominal and %s numerical features.', num_nominal_features,
                     (len(dataset.features) - num_nominal_features))

            if num_nominal_features > 0:
                x = dataset.x

                if is_sparse(x):
                    x = x.toarray()

                encoder = self.encoder

                if not encoder:
                    log.info('Applying one-hot encoding...')
                    encoder = ColumnTransformer(
                        [('one_hot_encoder', SkLearnOneHotEncoder(handle_unknown='ignore',
                                                                  sparse_output=False), nominal_indices)],
                        remainder='passthrough')
                    encoder.fit(x, dataset.y)
                    self.encoder = encoder

                x = encoder.transform(x)
                return replace(dataset, x=x, features=[])

            log.debug('No need to apply one-hot encoding, as the data set does not contain any nominal features.')
            return dataset

    def create_encoder(self) -> Preprocessor.Encoder:
        return OneHotEncoder.Encoder()
