"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from unittest import main

from integration_tests import IntegrationTests, CmdBuilder


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
        self.run_cmd(builder, 'boomer_evaluation_train_test')

    def test_evaluation_cross_validation(self):
        """
        Tests the evaluation of the BOOMER algorithm using a cross validation.
        """
        builder = CmdBuilder() \
            .cross_validation() \
            .set_output_dir() \
            .print_evaluation() \
            .store_evaluation()
        self.run_cmd(builder, 'boomer_evaluation_cross_validation')

    def test_evaluation_single_fold(self):
        """
        Tests the evaluation of the BOOMER algorithm when using a single fold of a cross validation.
        """
        builder = CmdBuilder() \
            .cross_validation(current_fold=1) \
            .set_output_dir() \
            .print_evaluation() \
            .store_evaluation()
        self.run_cmd(builder, 'boomer_evaluation_single_fold')

    def test_evaluation_training_data(self):
        """
        Tests the evaluation of the BOOMER algorithm on the training data.
        """
        builder = CmdBuilder() \
            .evaluate_training_data() \
            .set_output_dir() \
            .print_evaluation() \
            .store_evaluation()
        self.run_cmd(builder, 'boomer_evaluation_training_data')

    def test_model_persistence_train_test(self):
        """
        Tests the functionality to store BOOMER models and load them afterwards when using a predefined split of the
        dataset into training and test data.
        """
        builder = CmdBuilder() \
            .set_model_dir()
        self.run_cmd(builder, 'boomer_model_persistence_train_test')

    def test_model_persistence_cross_validation(self):
        """
        Tests the functionality to store BOOMER models and load them afterwards when using a cross validation.
        """
        builder = CmdBuilder() \
            .cross_validation() \
            .set_model_dir()
        self.run_cmd(builder, 'boomer_model_persistence_cross_validation')

    def test_model_persistence_single_fold(self):
        """
        Tests the functionality to store BOOMER models and load them afterwards when using a single fold of a cross
        validation.
        """
        builder = CmdBuilder() \
            .cross_validation(current_fold=1) \
            .set_model_dir()
        self.run_cmd(builder, 'boomer_model_persistence_single_fold')

    def test_predictions_train_test(self):
        """
        Tests the functionality to store the predictions of the BOOMER algorithm when using a predefined split of the
        dataset into training and test data.
        """
        builder = CmdBuilder() \
            .set_output_dir() \
            .print_predictions() \
            .store_predictions()
        self.run_cmd(builder, 'boomer_predictions_train_test')

    def test_predictions_cross_validation(self):
        """
        Tests the functionality to store the predictions of the BOOMER algorithm when using a cross validation.
        """
        builder = CmdBuilder() \
            .cross_validation() \
            .set_output_dir() \
            .print_predictions() \
            .store_predictions()
        self.run_cmd(builder, 'boomer_predictions_cross_validation')

    def test_predictions_single_fold(self):
        """
        Tests the functionality to store the predictions of the BOOMER algorithm when using a single fold of a cross
        validation.
        """
        builder = CmdBuilder() \
            .cross_validation(current_fold=1) \
            .set_output_dir() \
            .print_predictions() \
            .store_predictions()
        self.run_cmd(builder, 'boomer_predictions_single_fold')

    def test_predictions_training_data(self):
        """
        Tests the functionality to store the predictions of the BOOMER algorithm for the training data.
        """
        builder = CmdBuilder() \
            .evaluate_training_data() \
            .set_output_dir() \
            .print_predictions() \
            .store_predictions()
        self.run_cmd(builder, 'boomer_predictions_training_data')


if __name__ == '__main__':
    main()
