"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for preprocessing datasets.
"""
import logging as log

from dataclasses import replace
from typing import override

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder as SkLearnOneHotEncoder

from mlrl.testbed_sklearn.experiments.dataset import AttributeType

from mlrl.testbed.experiments.dataset import Dataset
from mlrl.testbed.experiments.input.dataset.preprocessors.preprocessor import Preprocessor


class OneHotEncoder(Preprocessor):
    """
    Allows one-hot-encoding all nominal features contained in a tabular dataset, if any.
    """

    class Encoder(Preprocessor.Encoder):
        """
        Allows one-hot-encoding all nominal features contained in a tabular dataset.
        """

        def __init__(self):
            self.encoder = None

        @override
        def encode(self, dataset: Dataset) -> Dataset:
            """
            See :func:`mlrl.testbed.experiments.input.dataset.preprocessors.Preprocessor.Encoder.encode`
            """
            nominal_indices = dataset.get_feature_indices(AttributeType.NOMINAL)
            num_nominal_features = len(nominal_indices)
            log.info('Dataset contains %s nominal and %s numerical features.', num_nominal_features,
                     (len(dataset.features) - num_nominal_features))

            if num_nominal_features > 0:
                dataset = dataset.enforce_dense_features()

                encoder = self.encoder

                if not encoder:
                    log.info('Applying one-hot encoding...')
                    one_hot_encoder = SkLearnOneHotEncoder(handle_unknown='ignore', sparse_output=False)
                    transformers = [('one_hot_encoder', one_hot_encoder, nominal_indices)]
                    encoder = ColumnTransformer(transformers, remainder='passthrough')
                    encoder.fit(dataset.x, dataset.y)
                    self.encoder = encoder

                return replace(dataset, x=encoder.transform(dataset.x), features=[])

            log.debug('No need to apply one-hot encoding, as the dataset does not contain any nominal features.')
            return dataset

    @override
    def create_encoder(self) -> Preprocessor.Encoder:
        """
        See :func:`mlrl.testbed.experiments.input.dataset.preprocessors.Preprocessor.create_encoder`
        """
        return OneHotEncoder.Encoder()
