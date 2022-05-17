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


if __name__ == '__main__':
    main()
