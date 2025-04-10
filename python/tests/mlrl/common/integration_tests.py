"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from abc import ABC
from sys import platform
from typing import Any
from unittest import SkipTest, TestCase

from .cmd_builder import CmdBuilder
from .cmd_runner import CmdRunner
from .datasets import Dataset


class IntegrationTests(TestCase, ABC):
    """
    Defines a series of integration tests for any type of rule learning algorithm.
    """

    def __init__(self,
                 dataset_default: str = Dataset.EMOTIONS,
                 dataset_numerical_sparse: str = Dataset.LANGLOG,
                 dataset_binary: str = Dataset.ENRON,
                 dataset_nominal: str = Dataset.EMOTIONS_NOMINAL,
                 dataset_ordinal: str = Dataset.EMOTIONS_ORDINAL,
                 dataset_single_output: str = Dataset.BREAST_CANCER,
                 dataset_meka: str = Dataset.MEKA,
                 methodName='runTest'):
        """
        :param dataset_default:             The name of the dataset that should be used by default
        :param dataset_numerical_sparse:    The name of a dataset with sparse numerical features
        :param dataset_binary:              The name of a dataset with binary features
        :param dataset_nominal:             The name of a dataset with nominal features
        :param dataset_ordinal:             The name of a dataset with ordinal features
        :param dataset_single_output:       The name of a dataset with a single target variable
        :param dataset_meka:                The name of a dataset in the MEKA format
        """
        super().__init__(methodName)
        self.dataset_default = dataset_default
        self.dataset_numerical_sparse = dataset_numerical_sparse
        self.dataset_binary = dataset_binary
        self.dataset_nominal = dataset_nominal
        self.dataset_ordinal = dataset_ordinal
        self.dataset_single_output = dataset_single_output
        self.dataset_meka = dataset_meka

    def _create_cmd_builder(self, dataset: str = Dataset.EMOTIONS) -> Any:
        """
        Must be implemented by subclasses in order to create an object of type `CmdBuilder` that allows to configure the
        command for running a rule learner.

        :param dataset: The dataset that should be used
        :return:        The object that has been created
        """
        raise NotImplementedError('Method _create_cmd_builder not implemented by test class')

    @classmethod
    def setUpClass(cls):
        """
        Sets up the test class.
        """
        if cls is IntegrationTests:
            raise SkipTest(cls.__name__ + ' is an abstract base class')
        if not platform.startswith('linux'):
            raise SkipTest('Integration tests are only supported on Linux')

        super().setUpClass()

    def test_single_output(self):
        """
        Tests the evaluation of the rule learning algorithm when predicting for a single output.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_single_output) \
            .print_evaluation()
        CmdRunner(self, builder).run('single-output')

    def test_sparse_feature_value(self):
        """
        Tests the training of the rule learning algorithm when using a custom value for the sparse elements in the
        feature matrix.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_numerical_sparse) \
            .sparse_feature_value(1.0)
        CmdRunner(self, builder).run('sparse-feature-value')

    def test_meka_format(self):
        """
        Tests the evaluation of the rule learning algorithm when using the MEKA data format.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_meka) \
            .print_evaluation(False)
        CmdRunner(self, builder).run('meka-format')

    def test_evaluation_no_data_split(self):
        """
        Tests the evaluation of the rule learning algorithm when not using a split of the dataset into training and test
        data.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_default) \
            .no_data_split() \
            .print_evaluation() \
            .store_evaluation()
        CmdRunner(self, builder).run('evaluation_no-data-split')

    def test_evaluation_train_test(self):
        """
        Tests the evaluation of the rule learning algorithm when using a predefined split of the dataset into training
        and test data.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_default) \
            .print_evaluation() \
            .store_evaluation()
        CmdRunner(self, builder).run('evaluation_train-test')

    def test_evaluation_train_test_predefined(self):
        """
        Tests the evaluation of the rule learning algorithm when using a predefined split of the dataset into training
        and test data, as provided by separate files.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_default + '-predefined') \
            .print_evaluation() \
            .store_evaluation()
        CmdRunner(self, builder).run('evaluation_train-test-predefined')

    def test_evaluation_cross_validation(self):
        """
        Tests the evaluation of the rule learning algorithm when using a cross validation.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_default) \
            .cross_validation() \
            .print_evaluation() \
            .store_evaluation()
        CmdRunner(self, builder).run('evaluation_cross-validation')

    def test_evaluation_cross_validation_predefined(self):
        """
        Tests the evaluation of the rule learning algorithm when using predefined splits of the dataset into individual
        cross validation folds, as provided by separate files.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_default + '-predefined') \
            .cross_validation() \
            .print_evaluation() \
            .store_evaluation()
        CmdRunner(self, builder).run('evaluation_cross-validation-predefined')

    def test_evaluation_single_fold(self):
        """
        Tests the evaluation of the rule learning algorithm when using a single fold of a cross validation.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_default) \
            .cross_validation(current_fold=1) \
            .print_evaluation() \
            .store_evaluation()
        CmdRunner(self, builder).run('evaluation_single-fold')

    def test_evaluation_training_data(self):
        """
        Tests the evaluation of the rule learning algorithm on the training data.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_default) \
            .evaluate_training_data() \
            .print_evaluation() \
            .store_evaluation()
        CmdRunner(self, builder).run('evaluation_training-data')

    def test_evaluation_incremental(self):
        """
        Tests the repeated evaluation of the model that is learned by a rule learning algorithm, using subsets of the
        induced rules with increasing size.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_default) \
            .incremental_evaluation() \
            .print_evaluation() \
            .store_evaluation()
        CmdRunner(self, builder).run('evaluation_incremental')

    def test_model_persistence_train_test(self):
        """
        Tests the functionality to store models and load them afterward when using a split of the dataset into training
        and test data.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_default) \
            .set_model_dir()
        CmdRunner(self, builder).run('model-persistence_train-test')

    def test_model_persistence_cross_validation(self):
        """
        Tests the functionality to store models and load them afterward when using a cross validation.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_default) \
            .cross_validation() \
            .set_model_dir()
        CmdRunner(self, builder).run('model-persistence_cross-validation')

    def test_model_persistence_single_fold(self):
        """
        Tests the functionality to store models and load them afterward when using a single fold of a cross validation.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_default) \
            .cross_validation(current_fold=1) \
            .set_model_dir()
        CmdRunner(self, builder).run('model-persistence_single-fold')

    def test_predictions_train_test(self):
        """
        Tests the functionality to store the predictions of the rule learning algorithm when using a split of the
        dataset into training and test data.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_default) \
            .print_evaluation(False) \
            .store_evaluation(False) \
            .print_predictions() \
            .store_predictions()
        CmdRunner(self, builder).run('predictions_train-test')

    def test_predictions_cross_validation(self):
        """
        Tests the functionality to store the predictions of the rule learning algorithm when using a cross validation.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_default) \
            .cross_validation() \
            .print_evaluation(False) \
            .store_evaluation(False) \
            .print_predictions() \
            .store_predictions()
        CmdRunner(self, builder).run('predictions_cross-validation')

    def test_predictions_single_fold(self):
        """
        Tests the functionality to store the predictions of the rule learning algorithm when using a single fold of a
        cross validation.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_default) \
            .cross_validation(current_fold=1) \
            .print_evaluation(False) \
            .store_evaluation(False) \
            .print_predictions() \
            .store_predictions()
        CmdRunner(self, builder).run('predictions_single-fold')

    def test_predictions_training_data(self):
        """
        Tests the functionality to store the predictions of the rule learning algorithm for the training data.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_default) \
            .evaluate_training_data() \
            .print_evaluation(False) \
            .store_evaluation(False) \
            .print_predictions() \
            .store_predictions()
        CmdRunner(self, builder).run('predictions_training-data')

    def test_prediction_characteristics_train_test(self):
        """
        Tests the functionality to store the prediction characteristics of the rule learning algorithm when using a
        split of the dataset into training and test data.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_default) \
            .print_evaluation(False) \
            .store_evaluation(False) \
            .print_prediction_characteristics() \
            .store_prediction_characteristics()
        CmdRunner(self, builder).run('prediction-characteristics_train-test')

    def test_prediction_characteristics_cross_validation(self):
        """
        Tests the functionality to store the prediction characteristics of the rule learning algorithm when using a
        cross validation.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_default) \
            .cross_validation() \
            .print_evaluation(False) \
            .store_evaluation(False) \
            .print_prediction_characteristics() \
            .store_prediction_characteristics()
        CmdRunner(self, builder).run('prediction-characteristics_cross-validation')

    def test_prediction_characteristics_single_fold(self):
        """
        Tests the functionality to store the prediction characteristics of the rule learning algorithm when using a
        single fold of a cross validation.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_default) \
            .cross_validation(current_fold=1) \
            .print_evaluation(False) \
            .store_evaluation(False) \
            .print_prediction_characteristics() \
            .store_prediction_characteristics()
        CmdRunner(self, builder).run('prediction-characteristics_single-fold')

    def test_prediction_characteristics_training_data(self):
        """
        Tests the functionality to store the prediction characteristics of the rule learning algorithm for the training
        data.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_default) \
            .evaluate_training_data() \
            .print_evaluation(False) \
            .store_evaluation(False) \
            .print_prediction_characteristics() \
            .store_prediction_characteristics()
        CmdRunner(self, builder).run('prediction-characteristics_training-data')

    def test_data_characteristics_train_test(self):
        """
        Tests the functionality to store the characteristics of the data used for training by the rule learning
        algorithm when using a split of the dataset into training and test data.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_default) \
            .print_evaluation(False) \
            .store_evaluation(False) \
            .print_data_characteristics() \
            .store_data_characteristics()
        CmdRunner(self, builder).run('data-characteristics_train-test')

    def test_data_characteristics_cross_validation(self):
        """
        Tests the functionality to store the characteristics of the data used for training by the rule learning
        algorithm when using a cross validation.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_default) \
            .cross_validation() \
            .print_evaluation(False) \
            .store_evaluation(False) \
            .print_data_characteristics() \
            .store_data_characteristics()
        CmdRunner(self, builder).run('data-characteristics_cross-validation')

    def test_data_characteristics_single_fold(self):
        """
        Tests the functionality to store the characteristics of the data used for training by the rule learning
        algorithm when using a single fold of a cross validation.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_default) \
            .cross_validation(current_fold=1) \
            .print_evaluation(False) \
            .store_evaluation(False) \
            .print_data_characteristics() \
            .store_data_characteristics()
        CmdRunner(self, builder).run('data-characteristics_single-fold')

    def test_model_characteristics_train_test(self):
        """
        Tests the functionality to store the characteristics of models when using a split of the dataset into training
        and test data.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_default) \
            .print_evaluation(False) \
            .store_evaluation(False) \
            .print_model_characteristics() \
            .store_model_characteristics()
        CmdRunner(self, builder).run('model-characteristics_train-test')

    def test_model_characteristics_cross_validation(self):
        """
        Tests the functionality to store the characteristics of models when using a cross validation.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_default) \
            .cross_validation() \
            .print_evaluation(False) \
            .store_evaluation(False) \
            .print_model_characteristics() \
            .store_model_characteristics()
        CmdRunner(self, builder).run('model-characteristics_cross-validation')

    def test_model_characteristics_single_fold(self):
        """
        Tests the functionality to store the characteristics of models when using a single fold of a cross validation.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_default) \
            .cross_validation(current_fold=1) \
            .print_evaluation(False) \
            .store_evaluation(False) \
            .print_model_characteristics() \
            .store_model_characteristics()
        CmdRunner(self, builder).run('model-characteristics_single-fold')

    def test_rules_train_test(self):
        """
        Tests the functionality to store textual representations of the rules in a model when using a split of the
        dataset into training and test data.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_default) \
            .print_evaluation(False) \
            .store_evaluation(False) \
            .print_rules() \
            .store_rules()
        CmdRunner(self, builder).run('rules_train-test')

    def test_rules_cross_validation(self):
        """
        Tests the functionality to store textual representations of the rules in a model when using a cross validation.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_default) \
            .cross_validation() \
            .print_evaluation(False) \
            .store_evaluation(False) \
            .print_rules() \
            .store_rules()
        CmdRunner(self, builder).run('rules_cross-validation')

    def test_rules_single_fold(self):
        """
        Tests the functionality to store textual representations of the rules in a model when using a single fold of a
        cross validation.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_default) \
            .cross_validation(current_fold=1) \
            .print_evaluation(False) \
            .store_evaluation(False) \
            .print_rules() \
            .store_rules()
        CmdRunner(self, builder).run('rules_single-fold')

    def test_numeric_features_dense(self):
        """
        Tests the rule learning algorithm on a dataset with numerical features when using a dense feature
        representation.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_numerical_sparse) \
            .sparse_feature_format(False)
        CmdRunner(self, builder).run('numeric-features-dense')

    def test_numeric_features_sparse(self):
        """
        Tests the rule learning algorithm on a dataset with numerical features when using a sparse feature
        representation.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_numerical_sparse) \
            .sparse_feature_format()
        CmdRunner(self, builder).run('numeric-features-sparse')

    def test_binary_features_dense(self):
        """
        Tests the rule learning algorithm on a dataset with binary features when using a dense feature representation.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_binary) \
            .sparse_feature_format(False)
        CmdRunner(self, builder).run('binary-features-dense')

    def test_binary_features_sparse(self):
        """
        Tests the rule learning algorithm on a dataset with binary features when using a sparse feature representation.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_binary) \
            .sparse_feature_format()
        CmdRunner(self, builder).run('binary-features-sparse')

    def test_nominal_features_dense(self):
        """
        Tests the rule learning algorithm on a dataset with nominal features when using a dense feature representation.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_nominal) \
            .sparse_feature_format(False)
        CmdRunner(self, builder).run('nominal-features-dense')

    def test_nominal_features_sparse(self):
        """
        Tests the rule learning algorithm on a dataset with nominal features when using a sparse feature representation.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_nominal) \
            .sparse_feature_format()
        CmdRunner(self, builder).run('nominal-features-sparse')

    def test_ordinal_features_dense(self):
        """
        Tests the rule learning algorithm on a dataset with ordinal features when using a dense feature representation.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_ordinal) \
            .sparse_feature_format(False)
        CmdRunner(self, builder).run('ordinal-features-dense')

    def test_ordinal_features_sparse(self):
        """
        Tests the rule learning algorithm on a dataset with ordinal features when using a sparse feature representation.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_ordinal) \
            .sparse_feature_format()
        CmdRunner(self, builder).run('ordinal-features-sparse')

    def test_output_format_dense(self):
        """
        Tests the rule learning algorithm when using a dense output representation.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_default) \
            .sparse_output_format(False)
        CmdRunner(self, builder).run('output-format-dense')

    def test_output_format_sparse(self):
        """
        Tests the rule learning algorithm when using a sparse output representation.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_default) \
            .sparse_output_format()
        CmdRunner(self, builder).run('output-format-sparse')

    def test_prediction_format_dense(self):
        """
        Tests the rule learning algorithm when using a dense representation of predictions.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_default) \
            .sparse_prediction_format(False) \
            .print_predictions()
        CmdRunner(self, builder).run('prediction-format-dense')

    def test_prediction_format_sparse(self):
        """
        Tests the rule learning algorithm when using a sparse representation of predictions.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_default) \
            .sparse_prediction_format() \
            .print_predictions()
        CmdRunner(self, builder).run('prediction-format-sparse')

    def test_parameters_train_test(self):
        """
        Tests the functionality to configure the rule learning algorithm according to parameter settings that are loaded
        from input files when using a split of the dataset into training and test data.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_default) \
            .print_evaluation(False) \
            .store_evaluation(False) \
            .print_model_characteristics() \
            .print_parameters() \
            .set_parameter_save_dir() \
            .set_parameter_load_dir()
        CmdRunner(self, builder).run('parameters_train-test')

    def test_parameters_cross_validation(self):
        """
        Tests the functionality to configure the rule learning algorithm according to parameter settings that are loaded
        from input files when using a cross validation.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_default) \
            .cross_validation() \
            .print_evaluation(False) \
            .store_evaluation(False) \
            .print_model_characteristics() \
            .print_parameters() \
            .set_parameter_save_dir() \
            .set_parameter_load_dir()
        CmdRunner(self, builder).run('parameters_cross-validation')

    def test_parameters_single_fold(self):
        """
        Tests the functionality to configure the rule learning algorithm according to parameter settings that are loaded
        from input files when using a single fold of a cross validation.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_default) \
            .cross_validation(current_fold=1) \
            .print_evaluation(False) \
            .store_evaluation(False) \
            .print_model_characteristics() \
            .print_parameters() \
            .set_parameter_save_dir() \
            .set_parameter_load_dir()
        CmdRunner(self, builder).run('parameters_single-fold')

    def test_instance_sampling_no(self):
        """
        Tests the rule learning algorithm when not using a method to sample from the available training examples.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_default) \
            .instance_sampling(CmdBuilder.INSTANCE_SAMPLING_NO)
        CmdRunner(self, builder).run('instance-sampling-no')

    def test_instance_sampling_with_replacement(self):
        """
        Tests the rule learning algorithm when using a method to sample from the available training examples with
        replacement.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_default) \
            .instance_sampling(CmdBuilder.INSTANCE_SAMPLING_WITH_REPLACEMENT)
        CmdRunner(self, builder).run('instance-sampling-with-replacement')

    def test_instance_sampling_without_replacement(self):
        """
        Tests the rule learning algorithm when using a method to sample from the available training examples without
        replacement.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_default) \
            .instance_sampling(CmdBuilder.INSTANCE_SAMPLING_WITHOUT_REPLACEMENT)
        CmdRunner(self, builder).run('instance-sampling-without-replacement')

    def test_feature_sampling_no(self):
        """
        Tests the rule learning algorithm when not using a method to sample from the available features.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_default) \
            .feature_sampling(CmdBuilder.FEATURE_SAMPLING_NO)
        CmdRunner(self, builder).run('feature-sampling-no')

    def test_feature_sampling_without_replacement(self):
        """
        Tests the rule learning algorithm when using a method to sample from the available features without replacement.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_default) \
            .feature_sampling(CmdBuilder.FEATURE_SAMPLING_WITHOUT_REPLACEMENT)
        CmdRunner(self, builder).run('feature-sampling-without-replacement')

    def test_output_sampling_no(self):
        """
        Tests the rule learning algorithm when not using a method to sample from the available outputs.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_default) \
            .output_sampling(CmdBuilder.OUTPUT_SAMPLING_NO)
        CmdRunner(self, builder).run('output-sampling-no')

    def test_output_sampling_round_robin(self):
        """
        Tests the rule learning algorithm when using a method that samples single outputs in a round-robin fashion.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_default) \
            .output_sampling(CmdBuilder.OUTPUT_SAMPLING_ROUND_ROBIN)
        CmdRunner(self, builder).run('output-sampling-round-robin')

    def test_output_sampling_without_replacement(self):
        """
        Tests the rule learning algorithm when using a method to sample from the available outputs without replacement.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_default) \
            .output_sampling(CmdBuilder.OUTPUT_SAMPLING_WITHOUT_REPLACEMENT)
        CmdRunner(self, builder).run('output-sampling-without-replacement')

    def test_pruning_no(self):
        """
        Tests the rule learning algorithm when not using a pruning method.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_default) \
            .rule_pruning(CmdBuilder.RULE_PRUNING_NO)
        CmdRunner(self, builder).run('pruning-no')

    def test_pruning_irep(self):
        """
        Tests the rule learning algorithm when using the IREP pruning method.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_default) \
            .instance_sampling() \
            .rule_pruning(CmdBuilder.RULE_PRUNING_IREP)
        CmdRunner(self, builder).run('pruning-irep')

    def test_rule_induction_top_down_beam_search(self):
        """
        Tests the rule learning algorithm when using a top-down beam search.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_default) \
            .rule_induction(CmdBuilder.RULE_INDUCTION_TOP_DOWN_BEAM_SEARCH)
        CmdRunner(self, builder).run('rule-induction-top-down-beam-search')

    def test_sequential_post_optimization(self):
        """
        Tests the rule learning algorithm when using sequential post-optimization.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_default) \
            .sequential_post_optimization()
        CmdRunner(self, builder).run('sequential-post-optimization')

    def test_feature_binning_equal_width_binary_features_dense(self):
        """
        Tests the rule learning algorithm's ability to use equal-width feature binning when applied to a dataset with
        binary features using a dense feature representation.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_binary) \
            .feature_binning(CmdBuilder.FEATURE_BINNING_EQUAL_WIDTH) \
            .sparse_feature_format(False)
        CmdRunner(self, builder).run('feature-binning-equal-width_binary-features-dense')

    def test_feature_binning_equal_width_binary_features_sparse(self):
        """
        Tests the rule learning algorithm's ability to use equal-width feature binning when applied to a dataset with
        binary features using a sparse feature representation.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_binary) \
            .feature_binning(CmdBuilder.FEATURE_BINNING_EQUAL_WIDTH) \
            .sparse_feature_format()
        CmdRunner(self, builder).run('feature-binning-equal-width_binary-features-sparse')

    def test_feature_binning_equal_width_nominal_features_dense(self):
        """
        Tests the rule learning algorithm's ability to use equal-width feature binning when applied to a dataset with
        nominal features using a dense feature representation.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_nominal) \
            .feature_binning(CmdBuilder.FEATURE_BINNING_EQUAL_WIDTH) \
            .sparse_feature_format(False)
        CmdRunner(self, builder).run('feature-binning-equal-width_nominal-features-dense')

    def test_feature_binning_equal_width_nominal_features_sparse(self):
        """
        Tests the rule learning algorithm's ability to use equal-width feature binning when applied to a dataset with
        nominal features using a sparse feature representation.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_nominal) \
            .feature_binning(CmdBuilder.FEATURE_BINNING_EQUAL_WIDTH) \
            .sparse_feature_format()
        CmdRunner(self, builder).run('feature-binning-equal-width_nominal-features-sparse')

    def test_feature_binning_equal_width_numerical_features_dense(self):
        """
        Tests the rule learning algorithm's ability to use equal-width feature binning when applied to a dataset with
        numerical features using a dense feature representation.
        """
        builder = self._create_cmd_builder() \
            .feature_binning(CmdBuilder.FEATURE_BINNING_EQUAL_WIDTH) \
            .sparse_feature_format(False)
        CmdRunner(self, builder).run('feature-binning-equal-width_numerical-features-dense')

    def test_feature_binning_equal_width_numerical_features_sparse(self):
        """
        Tests the rule learning algorithm's ability to use equal-width feature binning when applied to a dataset with
        numerical features using a sparse feature representation.
        """
        builder = self._create_cmd_builder() \
            .feature_binning(CmdBuilder.FEATURE_BINNING_EQUAL_WIDTH) \
            .sparse_feature_format()
        CmdRunner(self, builder).run('feature-binning-equal-width_numerical-features-sparse')

    def test_feature_binning_equal_frequency_binary_features_dense(self):
        """
        Tests the rule learning algorithm's ability to use equal-frequency feature binning when applied to a dataset
        with binary features using a dense feature representation.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_binary) \
            .feature_binning(CmdBuilder.FEATURE_BINNING_EQUAL_FREQUENCY) \
            .sparse_feature_format(False)
        CmdRunner(self, builder).run('feature-binning-equal-frequency_binary-features-dense')

    def test_feature_binning_equal_frequency_binary_features_sparse(self):
        """
        Tests the rule learning algorithm's ability to use equal-frequency feature binning when applied to a dataset
        with binary features using a sparse feature representation.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_binary) \
            .feature_binning(CmdBuilder.FEATURE_BINNING_EQUAL_FREQUENCY) \
            .sparse_feature_format()
        CmdRunner(self, builder).run('feature-binning-equal-frequency_binary-features-sparse')

    def test_feature_binning_equal_frequency_nominal_features_dense(self):
        """
        Tests the rule learning algorithm's ability to use equal-frequency feature binning when applied to a dataset
        with nominal features using a dense feature representation.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_nominal) \
            .feature_binning(CmdBuilder.FEATURE_BINNING_EQUAL_FREQUENCY) \
            .sparse_feature_format(False)
        CmdRunner(self, builder).run('feature-binning-equal-frequency_nominal-features-dense')

    def test_feature_binning_equal_frequency_nominal_features_sparse(self):
        """
        Tests the rule learning algorithm's ability to use equal-frequency feature binning when applied to a dataset
        with nominal features using a sparse feature representation.
        """
        builder = self._create_cmd_builder(dataset=self.dataset_nominal) \
            .feature_binning(CmdBuilder.FEATURE_BINNING_EQUAL_FREQUENCY) \
            .sparse_feature_format()
        CmdRunner(self, builder).run('feature-binning-equal-frequency_nominal-features-sparse')

    def test_feature_binning_equal_frequency_numerical_features_dense(self):
        """
        Tests the rule learning algorithm's ability to use equal-frequency feature binning when applied to a dataset
        with numerical features using a dense feature representation.
        """
        builder = self._create_cmd_builder() \
            .feature_binning(CmdBuilder.FEATURE_BINNING_EQUAL_FREQUENCY) \
            .sparse_feature_format(False)
        CmdRunner(self, builder).run('feature-binning-equal-frequency_numerical-features-dense')

    def test_feature_binning_equal_frequency_numerical_features_sparse(self):
        """
        Tests the rule learning algorithm's ability to use equal-frequency feature binning when applied to a dataset
        with numerical features using a sparse feature representation.
        """
        builder = self._create_cmd_builder() \
            .feature_binning(CmdBuilder.FEATURE_BINNING_EQUAL_FREQUENCY) \
            .sparse_feature_format()
        CmdRunner(self, builder).run('feature-binning-equal-frequency_numerical-features-sparse')
