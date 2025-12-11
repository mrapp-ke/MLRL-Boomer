"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from typing import Optional

from ..common.cmd_builder import CmdBuilder
from ..common.cmd_builder_classification import ClassificationCmdBuilder
from ..common.datasets import Dataset
from .cmd_builder import BoomerCmdBuilderMixin

from mlrl.common.config.parameters import BINNING_EQUAL_WIDTH

from mlrl.boosting.config.parameters import PROBABILITY_CALIBRATION_ISOTONIC

from mlrl.util.cli import AUTO
from mlrl.util.options import Options


class BoomerClassifierCmdBuilder(ClassificationCmdBuilder, BoomerCmdBuilderMixin):
    """
    A builder that allows to configure a command for running the BOOMER algorithm for classification problems.
    """

    def __init__(self, dataset: str = Dataset.EMOTIONS):
        super().__init__(expected_output_dir=CmdBuilder.EXPECTED_OUTPUT_DIR / 'boosting' / 'classification',
                         batch_config=CmdBuilder.CONFIG_DIR / 'boosting' / 'classification' / 'batch_config.yml',
                         runnable_module_name='mlrl.boosting',
                         dataset=dataset)

    def marginal_probability_calibration(self,
                                         probability_calibrator: Optional[str] = PROBABILITY_CALIBRATION_ISOTONIC):
        """
        Configures the algorithm to fit a model for the calibration of marginal probabilities.

        :param probability_calibrator:  The name of the method that should be used to fit a calibration model
        :return:                        The builder itself
        """
        if probability_calibrator:
            self.add_algorithmic_argument('--marginal-probability-calibration', probability_calibrator)

        return self

    def joint_probability_calibration(self, probability_calibrator: Optional[str] = PROBABILITY_CALIBRATION_ISOTONIC):
        """
        Configures the algorithm to fit a model for the calibration of joint probabilities.

        :param probability_calibrator:  The name of the method that should be used to fit a calibration model
        :return:                        The builder itself
        """
        if probability_calibrator:
            self.add_algorithmic_argument('--joint-probability-calibration', probability_calibrator)

        return self

    def binary_predictor(self, binary_predictor: Optional[str] = AUTO, options: Options = Options()):
        """
        Configures the algorithm to use a specific method for predicting binary labels.

        :param binary_predictor:    The name of the method that should be used for predicting binary labels
        :param options:             Options to be taken into account
        :return:                    The builder itself
        """
        if binary_predictor:
            self.add_algorithmic_argument('--binary-predictor', binary_predictor + (str(options) if options else ''))

        return self

    def probability_predictor(self, probability_predictor: Optional[str] = AUTO):
        """
        Configures the algorithm to use a specific method for predicting probabilities.

        :param probability_predictor:   The name of the method that should be used for predicting probabilities
        :return:                        The builder itself
        """
        if probability_predictor:
            self.add_algorithmic_argument('--probability-predictor', probability_predictor)

        return self

    def label_binning(self, label_binning: Optional[str] = BINNING_EQUAL_WIDTH):
        """
        Configures the algorithm to use a specific method for the assignment of labels to bins.

        :param label_binning:   The name of the method to be used
        :return:                The builder itself
        """
        if label_binning:
            self.add_algorithmic_argument('--label-binning', label_binning)

        return self
