"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from os import path
from typing import Optional

from ..common.cmd_builder_classification import ClassificationCmdBuilder
from ..common.datasets import Dataset
from .cmd_builder import BoomerCmdBuilderMixin


class BoomerClassifierCmdBuilder(ClassificationCmdBuilder, BoomerCmdBuilderMixin):
    """
    A builder that allows to configure a command for running the BOOMER algorithm for classification problems.
    """

    LOSS_LOGISTIC_DECOMPOSABLE = 'logistic-decomposable'

    LOSS_LOGISTIC_NON_DECOMPOSABLE = 'logistic-non-decomposable'

    LOSS_SQUARED_HINGE_DECOMPOSABLE = 'squared-hinge-decomposable'

    LOSS_SQUARED_HINGE_NON_DECOMPOSABLE = 'squared-hinge-non-decomposable'

    LABEL_BINNING_NO = 'none'

    LABEL_BINNING_EQUAL_WIDTH = 'equal-width'

    PROBABILITY_CALIBRATOR_ISOTONIC = 'isotonic'

    BINARY_PREDICTOR_AUTO = 'auto'

    BINARY_PREDICTOR_OUTPUT_WISE = 'output-wise'

    BINARY_PREDICTOR_OUTPUT_WISE_BASED_ON_PROBABILITIES = (BINARY_PREDICTOR_OUTPUT_WISE
                                                           + '{based_on_probabilities=true}')

    BINARY_PREDICTOR_EXAMPLE_WISE = 'example-wise'

    BINARY_PREDICTOR_EXAMPLE_WISE_BASED_ON_PROBABILITIES = (BINARY_PREDICTOR_EXAMPLE_WISE
                                                            + '{based_on_probabilities=true}')

    BINARY_PREDICTOR_GFM = 'gfm'

    PROBABILITY_PREDICTOR_AUTO = 'auto'

    PROBABILITY_PREDICTOR_OUTPUT_WISE = 'output-wise'

    PROBABILITY_PREDICTOR_MARGINALIZED = 'marginalized'

    def __init__(self, dataset: str = Dataset.EMOTIONS):
        super().__init__(expected_output_dir=path.join('boosting', 'classification'),
                         runnable_module_name='mlrl.boosting',
                         dataset=dataset)

    def marginal_probability_calibration(self, probability_calibrator: Optional[str] = PROBABILITY_CALIBRATOR_ISOTONIC):
        """
        Configures the algorithm to fit a model for the calibration of marginal probabilities.

        :param probability_calibrator:  The name of the method that should be used to fit a calibration model
        :return:                        The builder itself
        """
        if probability_calibrator:
            self.args.append('--marginal-probability-calibration')
            self.args.append(probability_calibrator)

        return self

    def joint_probability_calibration(self, probability_calibrator: Optional[str] = PROBABILITY_CALIBRATOR_ISOTONIC):
        """
        Configures the algorithm to fit a model for the calibration of joint probabilities.

        :param probability_calibrator:  The name of the method that should be used to fit a calibration model
        :return:                        The builder itself
        """
        if probability_calibrator:
            self.args.append('--joint-probability-calibration')
            self.args.append(probability_calibrator)

        return self

    def binary_predictor(self, binary_predictor: Optional[str] = BINARY_PREDICTOR_AUTO):
        """
        Configures the algorithm to use a specific method for predicting binary labels.

        :param binary_predictor:    The name of the method that should be used for predicting binary labels
        :return:                    The builder itself
        """
        if binary_predictor:
            self.args.append('--binary-predictor')
            self.args.append(binary_predictor)

        return self

    def probability_predictor(self, probability_predictor: Optional[str] = PROBABILITY_PREDICTOR_AUTO):
        """
        Configures the algorithm to use a specific method for predicting probabilities.

        :param probability_predictor:   The name of the method that should be used for predicting probabilities
        :return:                        The builder itself
        """
        if probability_predictor:
            self.args.append('--probability-predictor')
            self.args.append(probability_predictor)

        return self

    def label_binning(self, label_binning: Optional[str] = LABEL_BINNING_EQUAL_WIDTH):
        """
        Configures the algorithm to use a specific method for the assignment of labels to bins.

        :param label_binning:   The name of the method to be used
        :return:                The builder itself
        """
        if label_binning:
            self.args.append('--label-binning')
            self.args.append(label_binning)

        return self
