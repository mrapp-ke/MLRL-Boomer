"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)
"""

from os import path

from test_common import CommonIntegrationTests, CmdBuilder, DIR_OUT, DIR_DATA, DATASET_EMOTIONS, DATASET_WEATHER

CMD_SECO = 'seco'


class SeCoCmdBuilder(CmdBuilder):
    """
    A builder that allows to configure a command for running the separate-and-conquer (SeCo) algorithm.
    """

    def __init__(self, data_dir: str = DIR_DATA, dataset: str = DATASET_EMOTIONS):
        super(SeCoCmdBuilder, self).__init__(cmd=CMD_SECO, data_dir=data_dir, dataset=dataset)


class SeCoIntegrationTests(CommonIntegrationTests):
    """
    Defines a series of integration tests for the separate-and-conquer (SeCo) algorithm.
    """

    def __init__(self, methodName='runTest'):
        """
        :param methodName: The name of the test method to be executed
        """
        super(SeCoIntegrationTests, self).__init__(cmd=CMD_SECO, dataset_one_hot_encoding=DATASET_WEATHER,
                                                   expected_output_dir=path.join(DIR_OUT, CMD_SECO),
                                                   methodName=methodName)
