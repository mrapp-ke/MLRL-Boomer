"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from typing import Optional

from .cmd_builder import DATASET_EMOTIONS, CmdBuilder


class ClassificationCmdBuilder(CmdBuilder):
    """
    A builder that allows to configure a command for applying a rule learning algorithm to a classification problem.
    """

    PREDICTION_TYPE_BINARY = 'binary'

    PREDICTION_TYPE_SCORES = 'scores'

    PREDICTION_TYPE_PROBABILITIES = 'probabilities'

    INSTANCE_SAMPLING_STRATIFIED_OUTPUT_WISE = 'stratified-output-wise'

    INSTANCE_SAMPLING_STRATIFIED_EXAMPLE_WISE = 'stratified-example-wise'

    HOLDOUT_STRATIFIED_OUTPUT_WISE = 'stratified-output-wise'

    HOLDOUT_STRATIFIED_EXAMPLE_WISE = 'stratified-example-wise'

    def __init__(self,
                 expected_output_dir: str,
                 model_file_name: str,
                 runnable_module_name: str,
                 runnable_class_name: Optional[str] = None,
                 dataset: str = DATASET_EMOTIONS):
        super().__init__(expected_output_dir=expected_output_dir,
                         model_file_name=model_file_name,
                         runnable_module_name=runnable_module_name,
                         runnable_class_name=runnable_class_name,
                         dataset=dataset)
        self.label_vectors_stored = False
        self.marginal_probability_calibration_model_stored = False
        self.joint_probability_calibration_model_stored = False

    def print_label_vectors(self, print_label_vectors: bool = True):
        """
        Configures whether the unique label vectors contained in the training data should be printed on the console or
        not.

        :param print_label_vectors: True, if the unique label vectors contained in the training data should be printed,
                                    False otherwise
        :return:                    The builder itself    
        """
        self.args.append('--print-label-vectors')
        self.args.append(str(print_label_vectors).lower())
        return self

    def store_label_vectors(self, store_label_vectors: bool = True):
        """
        Configures whether the unique label vectors contained in the training data should be written into output files
        or not.

        :param store_label_vectors: True, if the unique label vectors contained in the training data should be written
                                    into output files, False otherwise
        :return:                    The builder itself
        """
        self.label_vectors_stored = store_label_vectors
        self.args.append('--store-label-vectors')
        self.args.append(str(store_label_vectors).lower())
        return self

    def print_marginal_probability_calibration_model(self, print_marginal_probability_calibration_model: bool = True):
        """
        Configures whether textual representations of models for the calibration of marginal probabilities should be
        printed on the console or not.

        :param print_marginal_probability_calibration_model:    True, if textual representations of models for the
                                                                calibration of marginal probabilities should be printed,
                                                                False otherwise
        :return:                                                The builder itself    
        """
        self.args.append('--print-marginal-probability-calibration-model')
        self.args.append(str(print_marginal_probability_calibration_model).lower())
        return self

    def store_marginal_probability_calibration_model(self, store_marginal_probability_calibration_model: bool = True):
        """
        Configures whether textual representations of models for the calibration of marginal probabilities should be
        written into output files or not.

        :param store_marginal_probability_calibration_model:    True, if textual representations of models for the
                                                                calibration of marginal probabilities should be written
                                                                into output files, False otherwise
        :return:                                                The builder itself    
        """
        self.marginal_probability_calibration_model_stored = store_marginal_probability_calibration_model
        self.args.append('--store-marginal-probability-calibration-model')
        self.args.append(str(store_marginal_probability_calibration_model).lower())
        return self

    def print_joint_probability_calibration_model(self, print_joint_probability_calibration_model: bool = True):
        """
        Configures whether textual representations of models for the calibration of joint probabilities should be
        printed on the console or not.

        :param print_joint_probability_calibration_model:   True, if textual representations of models for the
                                                            calibration of joint probabilities should be printed, False
                                                            otherwise
        :return:                                            The builder itself    
        """
        self.args.append('--print-joint-probability-calibration-model')
        self.args.append(str(print_joint_probability_calibration_model).lower())
        return self

    def store_joint_probability_calibration_model(self, store_joint_probability_calibration_model: bool = True):
        """
        Configures whether textual representations of models for the calibration of joint probabilities should be
        written into output files or not.

        :param store_joint_probability_calibration_model:   True, if textual representations of models for the
                                                            calibration of joint probabilities should be written into
                                                            output files, False otherwise
        :return:                                            The builder itself    
        """
        self.joint_probability_calibration_model_stored = store_joint_probability_calibration_model
        self.args.append('--store-joint-probability-calibration-model')
        self.args.append(str(store_joint_probability_calibration_model).lower())
        return self

    def prediction_type(self, prediction_type: str = PREDICTION_TYPE_BINARY):
        """
        Configures the type of predictions that should be obtained from the algorithm.

        :param prediction_type: The type of the predictions
        :return:                The builder itself
        """
        self.args.append('--prediction-type')
        self.args.append(prediction_type)
        return self
