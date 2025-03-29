"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for preprocessing datasets.
"""
import logging as log

from dataclasses import replace

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder as SkLearnOneHotEncoder

from mlrl.testbed.dataset import AttributeType, Dataset
from mlrl.testbed.experiments.input.preprocessors.preprocessor import Preprocessor


class OneHotEncoder(Preprocessor):
    """
    Allows one-hot-encoding all nominal features contained in a data set, if any.
    """

    class Encoder(Preprocessor.Encoder):
        """
        Allows one-hot-encoding all nominal features contained in a data set.
        """

        def __init__(self):
            self.encoder = None

        def encode(self, dataset: Dataset) -> Dataset:
            """
            See :func:`mlrl.testbed.experiments.input.preprocessors.Preprocessor.Encoder.encode`
            """
            nominal_indices = dataset.get_feature_indices(AttributeType.NOMINAL)
            num_nominal_features = len(nominal_indices)
            log.info('Data set contains %s nominal and %s numerical features.', num_nominal_features,
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

            log.debug('No need to apply one-hot encoding, as the data set does not contain any nominal features.')
            return dataset

    def create_encoder(self) -> Preprocessor.Encoder:
        """
        See :func:`mlrl.testbed.experiments.input.preprocessors.Preprocessor.create_encoder`
        """
        return OneHotEncoder.Encoder()
