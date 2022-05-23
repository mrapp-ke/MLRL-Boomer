"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from unittest import main

from integration_tests import IntegrationTests, CmdBuilder, DATASET_ENRON, DATASET_LANGLOG


class BoostingIntegrationTests(IntegrationTests):
    """
    Defines a series of integration tests that run the BOOMER algorithm.
    """

    def test_evaluation_train_test(self):
        """
        Tests the evaluation of the BOOMER algorithm when using a predefined split of the dataset into training and test
        data.
        """
        builder = CmdBuilder() \
            .set_output_dir() \
            .print_evaluation() \
            .store_evaluation()
        self.run_cmd(builder, 'boomer_evaluation_train-test')

    def test_evaluation_cross_validation(self):
        """
        Tests the evaluation of the BOOMER algorithm when using a cross validation.
        """
        builder = CmdBuilder() \
            .cross_validation() \
            .set_output_dir() \
            .print_evaluation() \
            .store_evaluation()
        self.run_cmd(builder, 'boomer_evaluation_cross-validation')

    def test_evaluation_single_fold(self):
        """
        Tests the evaluation of the BOOMER algorithm when using a single fold of a cross validation.
        """
        builder = CmdBuilder() \
            .cross_validation(current_fold=1) \
            .set_output_dir() \
            .print_evaluation() \
            .store_evaluation()
        self.run_cmd(builder, 'boomer_evaluation_single-fold')

    def test_evaluation_training_data(self):
        """
        Tests the evaluation of the BOOMER algorithm on the training data.
        """
        builder = CmdBuilder() \
            .evaluate_training_data() \
            .set_output_dir() \
            .print_evaluation() \
            .store_evaluation()
        self.run_cmd(builder, 'boomer_evaluation_training-data')

    def test_model_persistence_train_test(self):
        """
        Tests the functionality to store BOOMER models and load them afterwards when using a predefined split of the
        dataset into training and test data.
        """
        builder = CmdBuilder() \
            .set_model_dir()
        self.run_cmd(builder, 'boomer_model-persistence_train-test')

    def test_model_persistence_cross_validation(self):
        """
        Tests the functionality to store BOOMER models and load them afterwards when using a cross validation.
        """
        builder = CmdBuilder() \
            .cross_validation() \
            .set_model_dir()
        self.run_cmd(builder, 'boomer_model-persistence_cross-validation')

    def test_model_persistence_single_fold(self):
        """
        Tests the functionality to store BOOMER models and load them afterwards when using a single fold of a cross
        validation.
        """
        builder = CmdBuilder() \
            .cross_validation(current_fold=1) \
            .set_model_dir()
        self.run_cmd(builder, 'boomer_model-persistence_single-fold')

    def test_predictions_train_test(self):
        """
        Tests the functionality to store the predictions of the BOOMER algorithm when using a predefined split of the
        dataset into training and test data.
        """
        builder = CmdBuilder() \
            .print_evaluation(False) \
            .store_evaluation(False) \
            .set_output_dir() \
            .print_predictions() \
            .store_predictions()
        self.run_cmd(builder, 'boomer_predictions_train-test')

    def test_predictions_cross_validation(self):
        """
        Tests the functionality to store the predictions of the BOOMER algorithm when using a cross validation.
        """
        builder = CmdBuilder() \
            .cross_validation() \
            .print_evaluation(False) \
            .store_evaluation(False) \
            .set_output_dir() \
            .print_predictions() \
            .store_predictions()
        self.run_cmd(builder, 'boomer_predictions_cross-validation')

    def test_predictions_single_fold(self):
        """
        Tests the functionality to store the predictions of the BOOMER algorithm when using a single fold of a cross
        validation.
        """
        builder = CmdBuilder() \
            .cross_validation(current_fold=1) \
            .print_evaluation(False) \
            .store_evaluation(False) \
            .set_output_dir() \
            .print_predictions() \
            .store_predictions()
        self.run_cmd(builder, 'boomer_predictions_single-fold')

    def test_predictions_training_data(self):
        """
        Tests the functionality to store the predictions of the BOOMER algorithm for the training data.
        """
        builder = CmdBuilder() \
            .evaluate_training_data() \
            .print_evaluation(False) \
            .store_evaluation(False) \
            .set_output_dir() \
            .print_predictions() \
            .store_predictions()
        self.run_cmd(builder, 'boomer_predictions_training-data')

    def test_prediction_characteristics_train_test(self):
        """
        Tests the functionality to store the prediction characteristics of the BOOMER algorithm when using a predefined
        split of the dataset into training and test data.
        """
        builder = CmdBuilder() \
            .print_evaluation(False) \
            .store_evaluation(False) \
            .set_output_dir() \
            .print_prediction_characteristics() \
            .store_prediction_characteristics()
        self.run_cmd(builder, 'boomer_prediction-characteristics_train-test')

    def test_prediction_characteristics_cross_validation(self):
        """
        Tests the functionality to store the prediction characteristics of the BOOMER algorithm when using a cross
        validation.
        """
        builder = CmdBuilder() \
            .cross_validation() \
            .print_evaluation(False) \
            .store_evaluation(False) \
            .set_output_dir() \
            .print_prediction_characteristics() \
            .store_prediction_characteristics()
        self.run_cmd(builder, 'boomer_prediction-characteristics_cross-validation')

    def test_prediction_characteristics_single_fold(self):
        """
        Tests the functionality to store the prediction characteristics of the BOOMER algorithm when using a single fold
        of a cross validation.
        """
        builder = CmdBuilder() \
            .cross_validation(current_fold=1) \
            .print_evaluation(False) \
            .store_evaluation(False) \
            .set_output_dir() \
            .print_prediction_characteristics() \
            .store_prediction_characteristics()
        self.run_cmd(builder, 'boomer_prediction-characteristics_single-fold')

    def test_prediction_characteristics_training_data(self):
        """
        Tests the functionality to store the prediction characteristics of the BOOMER algorithm for the training data.
        """
        builder = CmdBuilder() \
            .evaluate_training_data() \
            .print_evaluation(False) \
            .store_evaluation(False) \
            .set_output_dir() \
            .print_prediction_characteristics() \
            .store_prediction_characteristics()
        self.run_cmd(builder, 'boomer_prediction-characteristics_training-data')

    def test_data_characteristics_train_test(self):
        """
        Tests the functionality to store the characteristics of the data used for training by the BOOMER algorithm when
        using a predefined split of the dataset into training and test data.
        """
        builder = CmdBuilder() \
            .print_evaluation(False) \
            .store_evaluation(False) \
            .set_output_dir() \
            .print_data_characteristics() \
            .store_data_characteristics()
        self.run_cmd(builder, 'boomer_data-characteristics_train-test')

    def test_data_characteristics_cross_validation(self):
        """
        Tests the functionality to store the characteristics of the data used for training by the BOOMER algorithm when
        using a cross validation.
        """
        builder = CmdBuilder() \
            .cross_validation() \
            .print_evaluation(False) \
            .store_evaluation(False) \
            .set_output_dir() \
            .print_data_characteristics() \
            .store_data_characteristics()
        self.run_cmd(builder, 'boomer_data-characteristics_cross-validation')

    def test_data_characteristics_single_fold(self):
        """
        Tests the functionality to store the characteristics of the data used for training by the BOOMER algorithm when
        using a single fold of a cross validation.
        """
        builder = CmdBuilder() \
            .cross_validation(current_fold=1) \
            .print_evaluation(False) \
            .store_evaluation(False) \
            .set_output_dir() \
            .print_data_characteristics() \
            .store_data_characteristics()
        self.run_cmd(builder, 'boomer_data-characteristics_single-fold')

    def test_model_characteristics_train_test(self):
        """
        Tests the functionality to store the characteristics of BOOMER models when using a predefined split of the
        dataset into training and test data.
        """
        builder = CmdBuilder() \
            .print_evaluation(False) \
            .store_evaluation(False) \
            .set_output_dir() \
            .print_model_characteristics() \
            .store_model_characteristics()
        self.run_cmd(builder, 'boomer_model-characteristics_train-test')

    def test_model_characteristics_cross_validation(self):
        """
        Tests the functionality to store the characteristics of BOOMER models when using a cross validation.
        """
        builder = CmdBuilder() \
            .cross_validation() \
            .print_evaluation(False) \
            .store_evaluation(False) \
            .set_output_dir() \
            .print_model_characteristics() \
            .store_model_characteristics()
        self.run_cmd(builder, 'boomer_model-characteristics_cross-validation')

    def test_model_characteristics_single_fold(self):
        """
        Tests the functionality to store the characteristics of BOOMER models when using a single fold of a cross
        validation.
        """
        builder = CmdBuilder() \
            .cross_validation(current_fold=1) \
            .print_evaluation(False) \
            .store_evaluation(False) \
            .set_output_dir() \
            .print_model_characteristics() \
            .store_model_characteristics()
        self.run_cmd(builder, 'boomer_model-characteristics_single-fold')

    def test_rules_train_test(self):
        """
        Tests the functionality to store textual representations of the rules in BOOMER models when using a predefined
        split of the dataset into training and test data.
        """
        builder = CmdBuilder() \
            .print_evaluation(False) \
            .store_evaluation(False) \
            .set_output_dir() \
            .print_rules() \
            .store_rules()
        self.run_cmd(builder, 'boomer_rules_train-test')

    def test_rules_cross_validation(self):
        """
        Tests the functionality to store textual representations of the rules in BOOMER models when using a cross
        validation.
        """
        builder = CmdBuilder() \
            .cross_validation() \
            .print_evaluation(False) \
            .store_evaluation(False) \
            .set_output_dir() \
            .print_rules() \
            .store_rules()
        self.run_cmd(builder, 'boomer_rules_cross-validation')

    def test_rules_single_fold(self):
        """
        Tests the functionality to store textual representations of the rules in BOOMER models when using a single fold
        of a cross validation.
        """
        builder = CmdBuilder() \
            .cross_validation(current_fold=1) \
            .print_evaluation(False) \
            .store_evaluation(False) \
            .set_output_dir() \
            .print_rules() \
            .store_rules()
        self.run_cmd(builder, 'boomer_rules_single-fold')

    def test_numeric_features_dense(self):
        """
        Tests the BOOMER algorithm on a dataset with numerical attributes when using a dense feature representation.
        """
        builder = CmdBuilder(dataset=DATASET_LANGLOG) \
            .sparse_feature_format(False)
        self.run_cmd(builder, 'boomer_numeric-features-dense')

    def test_numeric_features_sparse(self):
        """
        Tests the BOOMER algorithm on a dataset with numerical attributes when using a sparse feature representation.
        """
        builder = CmdBuilder(dataset=DATASET_LANGLOG) \
            .sparse_feature_format(True)
        self.run_cmd(builder, 'boomer_numeric-features-sparse')

    def test_nominal_features_dense(self):
        """
        Tests the BOOMER algorithm on a dataset with nominal attributes when using a dense feature representation.
        """
        builder = CmdBuilder(dataset=DATASET_ENRON) \
            .sparse_feature_format(False)
        self.run_cmd(builder, 'boomer_nominal-features-dense')

    def test_nominal_features_sparse(self):
        """
        Tests the BOOMER algorithm on a dataset with nominal attributes when using a sparse feature representation.
        """
        builder = CmdBuilder(dataset=DATASET_ENRON) \
            .sparse_feature_format(True)
        self.run_cmd(builder, 'boomer_nominal-features-sparse')

    def test_labels_dense(self):
        """
        Tests the BOOMER algorithm when using a dense label representation.
        """
        builder = CmdBuilder() \
            .sparse_label_format(False)
        self.run_cmd(builder, 'boomer_labels-dense')

    def test_labels_sparse(self):
        """
        Tests the BOOMER algorithm when using a sparse label representation.
        """
        builder = CmdBuilder() \
            .sparse_label_format(True)
        self.run_cmd(builder, 'boomer_labels-sparse')

    def test_one_hot_encoding_train_test(self):
        """
        Tests the BOOMER algorithm on a dataset with one-hot-encoded nominal attributes when using a predefined split of
        the dataset into training and test data.
        """
        builder = CmdBuilder(dataset=DATASET_ENRON) \
            .one_hot_encoding()
        self.run_cmd(builder, 'boomer_one-hot-encoding_train-test')

    def test_one_hot_encoding_cross_validation(self):
        """
        Tests the BOOMER algorithm on a dataset with one-hot-encoded nominal attributes when using a cross validation.
        """
        builder = CmdBuilder(dataset=DATASET_ENRON) \
            .cross_validation() \
            .one_hot_encoding()
        self.run_cmd(builder, 'boomer_one-hot-encoding_cross-validation')

    def test_parameters_train_test(self):
        """
        Tests the functionality to configure the BOOMER algorithm according to parameter settings that are loaded from
        input files when using a predefined split of the dataset into training and test data.
        """
        builder = CmdBuilder() \
            .print_evaluation(False) \
            .print_model_characteristics(True) \
            .set_parameter_dir()
        self.run_cmd(builder, 'boomer_parameters_train-test')

    def test_parameters_cross_validation(self):
        """
        Tests the functionality to configure the BOOMER algorithm according to parameter settings that are loaded from
        input files when using a cross validation.
        """
        builder = CmdBuilder() \
            .cross_validation() \
            .print_evaluation(False) \
            .print_model_characteristics(True) \
            .set_parameter_dir()
        self.run_cmd(builder, 'boomer_parameters_cross-validation')

    def test_parameters_single_fold(self):
        """
        Tests the functionality to configure the BOOMER algorithm according to parameter settings that are loaded from
        input files when using a single fold of a cross validation.
        """
        builder = CmdBuilder() \
            .cross_validation(current_fold=1) \
            .print_evaluation(False) \
            .print_model_characteristics(True) \
            .set_parameter_dir()
        self.run_cmd(builder, 'boomer_parameters_single-fold')


if __name__ == '__main__':
    main()
