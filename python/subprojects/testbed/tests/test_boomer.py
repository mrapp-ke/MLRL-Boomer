"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from unittest import main

from integration_tests import IntegrationTests, CmdBuilder


class BoostingIntegrationTests(IntegrationTests):
    """
    Defines a series of integration tests that run the boosting algorithm.
    """

    def test_train_test_split(self):
        """
        Tests the default configuration of the algorithm using a predefined split of the dataset into training and test
        data.
        """
        builder = CmdBuilder()
        self.run_cmd(builder, 'boomer_train_test')

    def test_cross_validation(self):
        """
        Tests the default configuration of the algorithm using cross validation.
        """
        builder = CmdBuilder() \
            .cross_validation()
        self.run_cmd(builder, 'boomer_cross_validation')

    def test_current_fold(self):
        """
        Tests the default configuration of the algorithm using a single fold of a cross validation.
        """
        builder = CmdBuilder() \
            .cross_validation(current_fold=1)
        self.run_cmd(builder, 'boomer_current_fold')

    def test_evaluate_training_data(self):
        """
        Tests the default configuration of the algorithm when evaluated on the training data.
        :return:
        """
        builder = CmdBuilder() \
            .evaluate_training_data()
        self.run_cmd(builder, 'boomer_evaluate_training_data')


if __name__ == '__main__':
    main()
