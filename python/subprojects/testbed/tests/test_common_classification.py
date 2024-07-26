"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from abc import ABC
from typing import Optional
from unittest import SkipTest

from test_common import DATASET_BREAST_CANCER, DATASET_EMOTIONS, DATASET_EMOTIONS_NOMINAL, DATASET_EMOTIONS_ORDINAL, \
    DATASET_ENRON, DATASET_LANGLOG, DIR_DATA, CmdBuilder, IntegrationTests

PREDICTION_TYPE_BINARY = 'binary'

PREDICTION_TYPE_SCORES = 'scores'

PREDICTION_TYPE_PROBABILITIES = 'probabilities'

INSTANCE_SAMPLING_STRATIFIED_OUTPUT_WISE = 'stratified-output-wise'

INSTANCE_SAMPLING_STRATIFIED_EXAMPLE_WISE = 'stratified-example-wise'

HOLDOUT_STRATIFIED_OUTPUT_WISE = 'stratified-output-wise'

HOLDOUT_STRATIFIED_EXAMPLE_WISE = 'stratified-example-wise'


class ClassificationCmdBuilder(CmdBuilder):
    """
    A builder that allows to configure a command for applying a rule learning algorithm to a classification problem.
    """

    def __init__(self,
                 callback: CmdBuilder.AssertionCallback,
                 expected_output_dir: str,
                 model_file_name: str,
                 runnable_module_name: str,
                 runnable_class_name: Optional[str] = None,
                 data_dir: str = DIR_DATA,
                 dataset: str = DATASET_EMOTIONS):
        super().__init__(callback=callback,
                         expected_output_dir=expected_output_dir,
                         model_file_name=model_file_name,
                         runnable_module_name=runnable_module_name,
                         runnable_class_name=runnable_class_name,
                         data_dir=data_dir,
                         dataset=dataset)
        self.label_vectors_stored = False
        self.marginal_probability_calibration_model_stored = False
        self.joint_probability_calibration_model_stored = False

    def __assert_label_vector_files_exist(self):
        """
        Asserts that the label vector files, which should be created by a command, exist.
        """
        if self.label_vectors_stored:
            self._assert_output_files_exist('label_vectors', 'csv')

    def __assert_marginal_probability_calibration_model_files_exist(self):
        """
        Asserts that the marginal probability calibration model files, which should be created by a command, exist.
        """
        if self.marginal_probability_calibration_model_stored:
            self._assert_output_files_exist('marginal_probability_calibration_model', 'csv')

    def __assert_joint_probability_calibration_model_files_exist(self):
        """
        Asserts that the joint probability calibration model files, which should be created by a command, exist.
        """
        if self.joint_probability_calibration_model_stored:
            self._assert_output_files_exist('joint_probability_calibration_model', 'csv')

    def _validate_output_files(self):
        super()._validate_output_files()
        self.__assert_label_vector_files_exist()
        self.__assert_marginal_probability_calibration_model_files_exist()
        self.__assert_joint_probability_calibration_model_files_exist()

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


class ClassificationIntegrationTests(IntegrationTests, ABC):
    """
    Defines a series of integration tests for any type of rule learning algorithm that can be applied to classification
    problems.
    """

    def __init__(self,
                 dataset_default: str = DATASET_EMOTIONS,
                 dataset_numerical_sparse: str = DATASET_LANGLOG,
                 dataset_binary: str = DATASET_ENRON,
                 dataset_nominal: str = DATASET_EMOTIONS_NOMINAL,
                 dataset_ordinal: str = DATASET_EMOTIONS_ORDINAL,
                 dataset_single_output: str = DATASET_BREAST_CANCER,
                 methodName='runTest'):
        super().__init__(dataset_default=dataset_default,
                         dataset_numerical_sparse=dataset_numerical_sparse,
                         dataset_binary=dataset_binary,
                         dataset_nominal=dataset_nominal,
                         dataset_ordinal=dataset_ordinal,
                         dataset_single_output=dataset_single_output,
                         methodName=methodName)

    @classmethod
    def setUpClass(cls):
        if cls is ClassificationIntegrationTests:
            raise SkipTest(cls.__name__ + ' is an abstract base class')

        super().setUpClass()

    def test_label_vectors_train_test(self):
        """
        Tests the functionality to store the unique label vectors contained in the data used for training by the rule
        learning algorithm when using a split of the dataset into training and test data.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_default) \
            .print_evaluation(False) \
            .store_evaluation(False) \
            .set_output_dir() \
            .print_label_vectors() \
            .store_label_vectors()
        builder.run_cmd('label-vectors_train-test')

    def test_label_vectors_cross_validation(self):
        """
        Tests the functionality to store the unique label vectors contained in the data used for training by the rule
        learning algorithm when using a cross validation.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_default) \
            .cross_validation() \
            .print_evaluation(False) \
            .store_evaluation(False) \
            .set_output_dir() \
            .print_label_vectors() \
            .store_label_vectors()
        builder.run_cmd('label-vectors_cross-validation')

    def test_label_vectors_single_fold(self):
        """
        Tests the functionality to store the unique label vectors contained in the data used for training by the rule
        learning algorithm when using a single fold of a cross validation.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_default) \
            .cross_validation(current_fold=1) \
            .print_evaluation(False) \
            .store_evaluation(False) \
            .set_output_dir() \
            .print_label_vectors() \
            .store_label_vectors()
        builder.run_cmd('label-vectors_single-fold')

    def test_instance_sampling_stratified_output_wise(self):
        """
        Tests the rule learning algorithm when using a method to sample from the available training examples using
        label-wise stratification.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_default) \
            .instance_sampling(INSTANCE_SAMPLING_STRATIFIED_OUTPUT_WISE)
        builder.run_cmd('instance-sampling-stratified-output-wise')

    def test_instance_sampling_stratified_example_wise(self):
        """
        Tests the rule learning algorithm when using a method to sample from the available training examples using
        example-wise stratification.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_default) \
            .instance_sampling(INSTANCE_SAMPLING_STRATIFIED_EXAMPLE_WISE)
        builder.run_cmd('instance-sampling-stratified-example-wise')
