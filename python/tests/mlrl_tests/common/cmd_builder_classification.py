"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from pathlib import Path
from typing import Optional

from .cmd_builder import CmdBuilder
from .datasets import Dataset

from mlrl.testbed.experiments.prediction_type import PredictionType


class ClassificationCmdBuilder(CmdBuilder):
    """
    A builder that allows to configure a command for applying a rule learning algorithm to a classification problem.
    """

    def __init__(self,
                 expected_output_dir: Path,
                 input_dir: Path,
                 batch_config: Path,
                 runnable_module_name: str,
                 runnable_class_name: Optional[str] = None,
                 dataset: str = Dataset.EMOTIONS):
        super().__init__(expected_output_dir=expected_output_dir,
                         input_dir=input_dir,
                         batch_config=batch_config,
                         runnable_module_name=runnable_module_name,
                         runnable_class_name=runnable_class_name,
                         dataset=dataset)
        self.label_vectors_stored = False
        self.marginal_probability_calibration_model_stored = False
        self.joint_probability_calibration_model_stored = False

    def print_label_vectors(self, print_label_vectors: Optional[bool] = True):
        """
        Configures whether the unique label vectors contained in the training data should be printed on the console or
        not.

        :param print_label_vectors: True, if the unique label vectors contained in the training data should be printed,
                                    False otherwise
        :return:                    The builder itself    
        """
        if print_label_vectors:
            self.add_control_argument('--print-label-vectors', str(print_label_vectors).lower())

        return self

    def save_label_vectors(self, save_label_vectors: bool = True):
        """
        Configures whether the unique label vectors contained in the training data should be written to output files
        or not.

        :param save_label_vectors:  True, if the unique label vectors contained in the training data should be written
                                    into output files, False otherwise
        :return:                    The builder itself
        """
        self.label_vectors_stored = save_label_vectors
        self.add_control_argument('--save-label-vectors', str(save_label_vectors).lower())
        return self

    def print_marginal_probability_calibration_model(self,
                                                     print_marginal_probability_calibration_model: Optional[bool] = True
                                                     ):
        """
        Configures whether textual representations of models for the calibration of marginal probabilities should be
        printed on the console or not.

        :param print_marginal_probability_calibration_model:    True, if textual representations of models for the
                                                                calibration of marginal probabilities should be printed,
                                                                False otherwise
        :return:                                                The builder itself    
        """
        if print_marginal_probability_calibration_model:
            self.add_control_argument('--print-marginal-probability-calibration-model',
                                      str(print_marginal_probability_calibration_model).lower())

        return self

    def save_marginal_probability_calibration_model(self,
                                                    save_marginal_probability_calibration_model: Optional[bool] = True):
        """
        Configures whether textual representations of models for the calibration of marginal probabilities should be
        written to output files or not.

        :param save_marginal_probability_calibration_model:    True, if textual representations of models for the
                                                                calibration of marginal probabilities should be written
                                                                into output files, False otherwise
        :return:                                                The builder itself    
        """
        if save_marginal_probability_calibration_model:
            self.marginal_probability_calibration_model_stored = save_marginal_probability_calibration_model
            self.add_control_argument('--save-marginal-probability-calibration-model',
                                      str(save_marginal_probability_calibration_model).lower())

        return self

    def print_joint_probability_calibration_model(self,
                                                  print_joint_probability_calibration_model: Optional[bool] = True):
        """
        Configures whether textual representations of models for the calibration of joint probabilities should be
        printed on the console or not.

        :param print_joint_probability_calibration_model:   True, if textual representations of models for the
                                                            calibration of joint probabilities should be printed, False
                                                            otherwise
        :return:                                            The builder itself    
        """
        if print_joint_probability_calibration_model:
            self.add_control_argument('--print-joint-probability-calibration-model',
                                      str(print_joint_probability_calibration_model).lower())

        return self

    def save_joint_probability_calibration_model(self, save_joint_probability_calibration_model: Optional[bool] = True):
        """
        Configures whether textual representations of models for the calibration of joint probabilities should be
        written to output files or not.

        :param save_joint_probability_calibration_model:    True, if textual representations of models for the
                                                            calibration of joint probabilities should be written to
                                                            output files, False otherwise
        :return:                                            The builder itself    
        """
        if save_joint_probability_calibration_model:
            self.joint_probability_calibration_model_stored = save_joint_probability_calibration_model
            self.add_control_argument('--save-joint-probability-calibration-model',
                                      str(save_joint_probability_calibration_model).lower())

        return self

    def prediction_type(self, prediction_type: Optional[str] = PredictionType.BINARY):
        """
        Configures the type of predictions that should be obtained from the algorithm.

        :param prediction_type: The type of the predictions
        :return:                The builder itself
        """
        if prediction_type:
            self.add_control_argument('--prediction-type', prediction_type)

        return self
