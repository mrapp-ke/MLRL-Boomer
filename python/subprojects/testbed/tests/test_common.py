"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from abc import ABC
from unittest import SkipTest

from integration_tests import IntegrationTests, CmdBuilder, DATASET_ENRON, DATASET_LANGLOG


class CommonIntegrationTests(IntegrationTests, ABC):
    """
    Defines a series of integration tests for any type of rule learning algorithm.
    """

    def __init__(self, cmd: str, methodName='runTest'):
        """
        :param cmd:         The command to be run by the integration tests
        :param methodName:  The name of the test method to be executed
        """
        super(CommonIntegrationTests, self).__init__(methodName)
        self.cmd = cmd

    @classmethod
    def setUpClass(cls):
        if cls is CommonIntegrationTests:
            raise SkipTest(cls.__name__ + ' is an abstract base class')
        else:
            super(CommonIntegrationTests, cls).setUpClass()

    def test_evaluation_train_test(self):
        """
        Tests the evaluation of the rule learning algorithm when using a predefined split of the dataset into training
        and test data.
        """
        builder = CmdBuilder(self.cmd) \
            .set_output_dir() \
            .print_evaluation() \
            .store_evaluation()
        self.run_cmd(builder, self.cmd + '_evaluation_train-test')

    def test_evaluation_cross_validation(self):
        """
        Tests the evaluation of the rule learning algorithm when using a cross validation.
        """
        builder = CmdBuilder(self.cmd) \
            .cross_validation() \
            .set_output_dir() \
            .print_evaluation() \
            .store_evaluation()
        self.run_cmd(builder, self.cmd + '_evaluation_cross-validation')

    def test_evaluation_single_fold(self):
        """
        Tests the evaluation of the rule learning algorithm when using a single fold of a cross validation.
        """
        builder = CmdBuilder(self.cmd) \
            .cross_validation(current_fold=1) \
            .set_output_dir() \
            .print_evaluation() \
            .store_evaluation()
        self.run_cmd(builder, self.cmd + '_evaluation_single-fold')

    def test_evaluation_training_data(self):
        """
        Tests the evaluation of the rule learning algorithm on the training data.
        """
        builder = CmdBuilder(self.cmd) \
            .evaluate_training_data() \
            .set_output_dir() \
            .print_evaluation() \
            .store_evaluation()
        self.run_cmd(builder, self.cmd + '_evaluation_training-data')

    def test_model_persistence_train_test(self):
        """
        Tests the functionality to store models and load them afterward when using a predefined split of the dataset
        into training and test data.
        """
        builder = CmdBuilder(self.cmd) \
            .set_model_dir()
        self.run_cmd(builder, self.cmd + '_model-persistence_train-test')

    def test_model_persistence_cross_validation(self):
        """
        Tests the functionality to store models and load them afterward when using a cross validation.
        """
        builder = CmdBuilder(self.cmd) \
            .cross_validation() \
            .set_model_dir()
        self.run_cmd(builder, self.cmd + '_model-persistence_cross-validation')

    def test_model_persistence_single_fold(self):
        """
        Tests the functionality to store models and load them afterward when using a single fold of a cross validation.
        """
        builder = CmdBuilder(self.cmd) \
            .cross_validation(current_fold=1) \
            .set_model_dir()
        self.run_cmd(builder, self.cmd + '_model-persistence_single-fold')

    def test_predictions_train_test(self):
        """
        Tests the functionality to store the predictions of the rule learning algorithm when using a predefined split of
        the dataset into training and test data.
        """
        builder = CmdBuilder(self.cmd) \
            .print_evaluation(False) \
            .store_evaluation(False) \
            .set_output_dir() \
            .print_predictions() \
            .store_predictions()
        self.run_cmd(builder, self.cmd + '_predictions_train-test')

    def test_predictions_cross_validation(self):
        """
        Tests the functionality to store the predictions of the rule learning algorithm when using a cross validation.
        """
        builder = CmdBuilder(self.cmd) \
            .cross_validation() \
            .print_evaluation(False) \
            .store_evaluation(False) \
            .set_output_dir() \
            .print_predictions() \
            .store_predictions()
        self.run_cmd(builder, self.cmd + '_predictions_cross-validation')

    def test_predictions_single_fold(self):
        """
        Tests the functionality to store the predictions of the rule learning algorithm when using a single fold of a
        cross validation.
        """
        builder = CmdBuilder(self.cmd) \
            .cross_validation(current_fold=1) \
            .print_evaluation(False) \
            .store_evaluation(False) \
            .set_output_dir() \
            .print_predictions() \
            .store_predictions()
        self.run_cmd(builder, self.cmd + '_predictions_single-fold')

    def test_predictions_training_data(self):
        """
        Tests the functionality to store the predictions of the rule learning algorithm for the training data.
        """
        builder = CmdBuilder(self.cmd) \
            .evaluate_training_data() \
            .print_evaluation(False) \
            .store_evaluation(False) \
            .set_output_dir() \
            .print_predictions() \
            .store_predictions()
        self.run_cmd(builder, self.cmd + '_predictions_training-data')

    def test_prediction_characteristics_train_test(self):
        """
        Tests the functionality to store the prediction characteristics of the rule learning algorithm when using a
        predefined split of the dataset into training and test data.
        """
        builder = CmdBuilder(self.cmd) \
            .print_evaluation(False) \
            .store_evaluation(False) \
            .set_output_dir() \
            .print_prediction_characteristics() \
            .store_prediction_characteristics()
        self.run_cmd(builder, self.cmd + '_prediction-characteristics_train-test')

    def test_prediction_characteristics_cross_validation(self):
        """
        Tests the functionality to store the prediction characteristics of the rule learning algorithm when using a
        cross validation.
        """
        builder = CmdBuilder(self.cmd) \
            .cross_validation() \
            .print_evaluation(False) \
            .store_evaluation(False) \
            .set_output_dir() \
            .print_prediction_characteristics() \
            .store_prediction_characteristics()
        self.run_cmd(builder, self.cmd + '_prediction-characteristics_cross-validation')

    def test_prediction_characteristics_single_fold(self):
        """
        Tests the functionality to store the prediction characteristics of the rule learning algorithm when using a
        single fold of a cross validation.
        """
        builder = CmdBuilder(self.cmd) \
            .cross_validation(current_fold=1) \
            .print_evaluation(False) \
            .store_evaluation(False) \
            .set_output_dir() \
            .print_prediction_characteristics() \
            .store_prediction_characteristics()
        self.run_cmd(builder, self.cmd + '_prediction-characteristics_single-fold')

    def test_prediction_characteristics_training_data(self):
        """
        Tests the functionality to store the prediction characteristics of the rule learning algorithm for the training
        data.
        """
        builder = CmdBuilder(self.cmd) \
            .evaluate_training_data() \
            .print_evaluation(False) \
            .store_evaluation(False) \
            .set_output_dir() \
            .print_prediction_characteristics() \
            .store_prediction_characteristics()
        self.run_cmd(builder, self.cmd + '_prediction-characteristics_training-data')

    def test_data_characteristics_train_test(self):
        """
        Tests the functionality to store the characteristics of the data used for training by the rule learning
        algorithm when using a predefined split of the dataset into training and test data.
        """
        builder = CmdBuilder(self.cmd) \
            .print_evaluation(False) \
            .store_evaluation(False) \
            .set_output_dir() \
            .print_data_characteristics() \
            .store_data_characteristics()
        self.run_cmd(builder, self.cmd + '_data-characteristics_train-test')

    def test_data_characteristics_cross_validation(self):
        """
        Tests the functionality to store the characteristics of the data used for training by the rule learning
        algorithm when using a cross validation.
        """
        builder = CmdBuilder(self.cmd) \
            .cross_validation() \
            .print_evaluation(False) \
            .store_evaluation(False) \
            .set_output_dir() \
            .print_data_characteristics() \
            .store_data_characteristics()
        self.run_cmd(builder, self.cmd + '_data-characteristics_cross-validation')

    def test_data_characteristics_single_fold(self):
        """
        Tests the functionality to store the characteristics of the data used for training by the rule learning
        algorithm when using a single fold of a cross validation.
        """
        builder = CmdBuilder(self.cmd) \
            .cross_validation(current_fold=1) \
            .print_evaluation(False) \
            .store_evaluation(False) \
            .set_output_dir() \
            .print_data_characteristics() \
            .store_data_characteristics()
        self.run_cmd(builder, self.cmd + '_data-characteristics_single-fold')

    def test_model_characteristics_train_test(self):
        """
        Tests the functionality to store the characteristics of models when using a predefined split of the dataset into
        training and test data.
        """
        builder = CmdBuilder(self.cmd) \
            .print_evaluation(False) \
            .store_evaluation(False) \
            .set_output_dir() \
            .print_model_characteristics() \
            .store_model_characteristics()
        self.run_cmd(builder, self.cmd + '_model-characteristics_train-test')

    def test_model_characteristics_cross_validation(self):
        """
        Tests the functionality to store the characteristics of models when using a cross validation.
        """
        builder = CmdBuilder(self.cmd) \
            .cross_validation() \
            .print_evaluation(False) \
            .store_evaluation(False) \
            .set_output_dir() \
            .print_model_characteristics() \
            .store_model_characteristics()
        self.run_cmd(builder, self.cmd + '_model-characteristics_cross-validation')

    def test_model_characteristics_single_fold(self):
        """
        Tests the functionality to store the characteristics of models when using a single fold of a cross validation.
        """
        builder = CmdBuilder(self.cmd) \
            .cross_validation(current_fold=1) \
            .print_evaluation(False) \
            .store_evaluation(False) \
            .set_output_dir() \
            .print_model_characteristics() \
            .store_model_characteristics()
        self.run_cmd(builder, self.cmd + '_model-characteristics_single-fold')

    def test_rules_train_test(self):
        """
        Tests the functionality to store textual representations of the rules in a model when using a predefined split
        of the dataset into training and test data.
        """
        builder = CmdBuilder(self.cmd) \
            .print_evaluation(False) \
            .store_evaluation(False) \
            .set_output_dir() \
            .print_rules() \
            .store_rules()
        self.run_cmd(builder, self.cmd + '_rules_train-test')

    def test_rules_cross_validation(self):
        """
        Tests the functionality to store textual representations of the rules in a model when using a cross validation.
        """
        builder = CmdBuilder(self.cmd) \
            .cross_validation() \
            .print_evaluation(False) \
            .store_evaluation(False) \
            .set_output_dir() \
            .print_rules() \
            .store_rules()
        self.run_cmd(builder, self.cmd + '_rules_cross-validation')

    def test_rules_single_fold(self):
        """
        Tests the functionality to store textual representations of the rules in a model when using a single fold of a
        cross validation.
        """
        builder = CmdBuilder(self.cmd) \
            .cross_validation(current_fold=1) \
            .print_evaluation(False) \
            .store_evaluation(False) \
            .set_output_dir() \
            .print_rules() \
            .store_rules()
        self.run_cmd(builder, self.cmd + '_rules_single-fold')

    def test_numeric_features_dense(self):
        """
        Tests the rule learning algorithm on a dataset with numerical attributes when using a dense feature
        representation.
        """
        builder = CmdBuilder(self.cmd, dataset=DATASET_LANGLOG) \
            .sparse_feature_format(False)
        self.run_cmd(builder, self.cmd + '_numeric-features-dense')

    def test_numeric_features_sparse(self):
        """
        Tests the rule learning algorithm on a dataset with numerical attributes when using a sparse feature
        representation.
        """
        builder = CmdBuilder(self.cmd, dataset=DATASET_LANGLOG) \
            .sparse_feature_format(True)
        self.run_cmd(builder, self.cmd + '_numeric-features-sparse')

    def test_nominal_features_dense(self):
        """
        Tests the rule learning algorithm on a dataset with nominal attributes when using a dense feature
        representation.
        """
        builder = CmdBuilder(self.cmd, dataset=DATASET_ENRON) \
            .sparse_feature_format(False)
        self.run_cmd(builder, self.cmd + '_nominal-features-dense')

    def test_nominal_features_sparse(self):
        """
        Tests the rule learning algorithm on a dataset with nominal attributes when using a sparse feature
        representation.
        """
        builder = CmdBuilder(self.cmd, dataset=DATASET_ENRON) \
            .sparse_feature_format(True)
        self.run_cmd(builder, self.cmd + '_nominal-features-sparse')

    def test_labels_dense(self):
        """
        Tests the rule learning algorithm when using a dense label representation.
        """
        builder = CmdBuilder(self.cmd) \
            .sparse_label_format(False)
        self.run_cmd(builder, self.cmd + '_labels-dense')

    def test_labels_sparse(self):
        """
        Tests the rule learning algorithm when using a sparse label representation.
        """
        builder = CmdBuilder(self.cmd) \
            .sparse_label_format(True)
        self.run_cmd(builder, self.cmd + '_labels-sparse')

    def test_one_hot_encoding_train_test(self):
        """
        Tests the rule learning algorithm on a dataset with one-hot-encoded nominal attributes when using a predefined
        split of the dataset into training and test data.
        """
        builder = CmdBuilder(self.cmd, dataset=DATASET_ENRON) \
            .one_hot_encoding()
        self.run_cmd(builder, self.cmd + '_one-hot-encoding_train-test')

    def test_one_hot_encoding_cross_validation(self):
        """
        Tests the rule learning algorithm on a dataset with one-hot-encoded nominal attributes when using a cross
        validation.
        """
        builder = CmdBuilder(self.cmd, dataset=DATASET_ENRON) \
            .cross_validation() \
            .one_hot_encoding()
        self.run_cmd(builder, self.cmd + '_one-hot-encoding_cross-validation')

    def test_parameters_train_test(self):
        """
        Tests the functionality to configure the rule learning algorithm according to parameter settings that are loaded
        from input files when using a predefined split of the dataset into training and test data.
        """
        builder = CmdBuilder(self.cmd) \
            .print_evaluation(False) \
            .print_model_characteristics(True) \
            .set_parameter_dir()
        self.run_cmd(builder, self.cmd + '_parameters_train-test')

    def test_parameters_cross_validation(self):
        """
        Tests the functionality to configure the rule learning algorithm according to parameter settings that are loaded
        from input files when using a cross validation.
        """
        builder = CmdBuilder(self.cmd) \
            .cross_validation() \
            .print_evaluation(False) \
            .print_model_characteristics(True) \
            .set_parameter_dir()
        self.run_cmd(builder, self.cmd + '_parameters_cross-validation')

    def test_parameters_single_fold(self):
        """
        Tests the functionality to configure the rule learning algorithm according to parameter settings that are loaded
        from input files when using a single fold of a cross validation.
        """
        builder = CmdBuilder(self.cmd) \
            .cross_validation(current_fold=1) \
            .print_evaluation(False) \
            .print_model_characteristics(True) \
            .set_parameter_dir()
        self.run_cmd(builder, self.cmd + '_parameters_single-fold')
