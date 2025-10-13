"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Defines command line arguments for configuring the functionality to predict for different datasets.
"""
from mlrl.util.cli import BoolArgument


class PredictionDatasetArguments:
    """
    Defines command line arguments for configuring the functionality to predict for different datasets.
    """

    PREDICT_FOR_TRAINING_DATA = BoolArgument(
        '--predict-for-training-data',
        default=False,
        description='Whether predictions should be obtained for the training data or not.',
    )

    PREDICT_FOR_TEST_DATA = BoolArgument(
        '--predict-for-test-data',
        default=True,
        description='Whether predictions should be obtained for the test data or not.',
    )
