"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)
"""

from os import path

from test_common import CommonIntegrationTests, CmdBuilder, DIR_OUT, DIR_DATA, DATASET_EMOTIONS, DATASET_WEATHER

CMD_SECO = 'seco'

HEURISTIC_ACCURACY = 'accuracy'

HEURISTIC_PRECISION = 'precision'

HEURISTIC_LAPLACE = 'laplace'

HEURISTIC_RECALL = 'recall'

HEURISTIC_WRA = 'weighted-relative-accuracy'

HEURISTIC_F_MEASURE = 'f-measure'

HEURISTIC_M_ESTIMATE = 'm-estimate'


class SeCoCmdBuilder(CmdBuilder):
    """
    A builder that allows to configure a command for running the separate-and-conquer (SeCo) algorithm.
    """

    def __init__(self, data_dir: str = DIR_DATA, dataset: str = DATASET_EMOTIONS):
        super(SeCoCmdBuilder, self).__init__(cmd=CMD_SECO, data_dir=data_dir, dataset=dataset)

    def heuristic(self, heuristic: str = HEURISTIC_F_MEASURE):
        """
        Configures the algorithm to use a specific heuristic for learning rules.

        :param heuristic:   The name of the heuristic that should be used for learning rules
        :return:            The builder itself
        """
        self.args.append('--heuristic')
        self.args.append(heuristic)
        return self

    def pruning_heuristic(self, heuristic: str = HEURISTIC_ACCURACY):
        """
        Configures the algorithm to use a specific heuristic for pruning rules.

        :param heuristic:   The name of the heuristic that should be used for pruning rules
        :return:            The builder itself
        """
        self.args.append('--pruning-heuristic')
        self.args.append(heuristic)
        return self


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
