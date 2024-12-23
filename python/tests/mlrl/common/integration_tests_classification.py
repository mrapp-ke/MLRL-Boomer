"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from abc import ABC
from unittest import SkipTest

from .cmd_builder import DATASET_BREAST_CANCER, DATASET_EMOTIONS, DATASET_EMOTIONS_NOMINAL, DATASET_EMOTIONS_ORDINAL, \
    DATASET_ENRON, DATASET_LANGLOG
from .cmd_builder_classification import INSTANCE_SAMPLING_STRATIFIED_EXAMPLE_WISE, \
    INSTANCE_SAMPLING_STRATIFIED_OUTPUT_WISE
from .integration_tests import IntegrationTests


class ClassificationIntegrationTests(IntegrationTests, ABC):
    """
    Defines a series of integration tests for any type of rule learning algorithm that can be applied to classification
    problems.
    """

    # pylint: disable=invalid-name
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
