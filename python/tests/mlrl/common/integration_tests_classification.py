"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from abc import ABC
from unittest import SkipTest

from .cmd_builder_classification import ClassificationCmdBuilder
from .cmd_runner import CmdRunner
from .datasets import Dataset
from .integration_tests import IntegrationTests


class ClassificationIntegrationTests(IntegrationTests, ABC):
    """
    Defines a series of integration tests for any type of rule learning algorithm that can be applied to classification
    problems.
    """

    # pylint: disable=invalid-name
    def __init__(self,
                 dataset_default: str = Dataset.EMOTIONS,
                 dataset_numerical_sparse: str = Dataset.LANGLOG,
                 dataset_binary: str = Dataset.ENRON,
                 dataset_nominal: str = Dataset.EMOTIONS_NOMINAL,
                 dataset_ordinal: str = Dataset.EMOTIONS_ORDINAL,
                 dataset_single_output: str = Dataset.BREAST_CANCER,
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
        """
        Sets up the test class.
        """
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
            .print_label_vectors() \
            .store_label_vectors()
        CmdRunner(self, builder).run('label-vectors_train-test')

    def test_label_vectors_cross_validation(self):
        """
        Tests the functionality to store the unique label vectors contained in the data used for training by the rule
        learning algorithm when using a cross validation.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_default) \
            .cross_validation() \
            .print_evaluation(False) \
            .store_evaluation(False) \
            .print_label_vectors() \
            .store_label_vectors()
        CmdRunner(self, builder).run('label-vectors_cross-validation')

    def test_label_vectors_single_fold(self):
        """
        Tests the functionality to store the unique label vectors contained in the data used for training by the rule
        learning algorithm when using a single fold of a cross validation.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_default) \
            .cross_validation(current_fold=1) \
            .print_evaluation(False) \
            .store_evaluation(False) \
            .print_label_vectors() \
            .store_label_vectors()
        CmdRunner(self, builder).run('label-vectors_single-fold')

    def test_instance_sampling_stratified_output_wise(self):
        """
        Tests the rule learning algorithm when using a method to sample from the available training examples using
        label-wise stratification.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_default) \
            .instance_sampling(ClassificationCmdBuilder.INSTANCE_SAMPLING_STRATIFIED_OUTPUT_WISE)
        CmdRunner(self, builder).run('instance-sampling-stratified-output-wise')

    def test_instance_sampling_stratified_example_wise(self):
        """
        Tests the rule learning algorithm when using a method to sample from the available training examples using
        example-wise stratification.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_default) \
            .instance_sampling(ClassificationCmdBuilder.INSTANCE_SAMPLING_STRATIFIED_EXAMPLE_WISE)
        CmdRunner(self, builder).run('instance-sampling-stratified-example-wise')
