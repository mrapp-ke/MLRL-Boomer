"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)
"""

from os import path

from test_common import CommonIntegrationTests, DIR_OUT, DATASET_WEATHER

CMD_SECO = 'seco'


class SeCoIntegrationTests(CommonIntegrationTests):
    """
    Defines a series of integration tests for the separate-and-conquer (SeCo) algorithm.
    """

    def __init__(self, methodName='runTest'):
        """
        :param methodName: The name of the test method to be executed
        """
        super(SeCoIntegrationTests, self).__init__(CMD_SECO, dataset_one_hot_encoding=DATASET_WEATHER,
                                                   expected_output_dir=path.join(DIR_OUT, CMD_SECO),
                                                   methodName=methodName)
