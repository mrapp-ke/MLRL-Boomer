"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from os import path

from test_common import CommonIntegrationTests, CmdBuilder, DIR_OUT, DIR_DATA, DATASET_EMOTIONS, DATASET_WEATHER, \
    PRUNING_IREP

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

    def test_heuristic_accuracy(self):
        """
        Tests the SeCo algorithm when using the accuracy heuristic for learning rules.
        """
        builder = SeCoCmdBuilder() \
            .heuristic(HEURISTIC_ACCURACY)
        self.run_cmd(builder, 'heuristic_accuracy')

    def test_heuristic_precision(self):
        """
        Tests the SeCo algorithm when using the precision heuristic for learning rules.
        """
        builder = SeCoCmdBuilder() \
            .heuristic(HEURISTIC_PRECISION)
        self.run_cmd(builder, 'heuristic_precision')

    def test_heuristic_recall(self):
        """
        Tests the SeCo algorithm when using the recall heuristic for learning rules.
        """
        builder = SeCoCmdBuilder() \
            .heuristic(HEURISTIC_RECALL)
        self.run_cmd(builder, 'heuristic_recall')

    def test_heuristic_laplace(self):
        """
        Tests the SeCo algorithm when using the Laplace heuristic for learning rules.
        """
        builder = SeCoCmdBuilder() \
            .heuristic(HEURISTIC_LAPLACE)
        self.run_cmd(builder, 'heuristic_laplace')

    def test_heuristic_wra(self):
        """
        Tests the SeCo algorithm when using the WRA heuristic for learning rules.
        """
        builder = SeCoCmdBuilder() \
            .heuristic(HEURISTIC_WRA)
        self.run_cmd(builder, 'heuristic_wra')

    def test_heuristic_f_measure(self):
        """
        Tests the SeCo algorithm when using the F-measure heuristic for learning rules.
        """
        builder = SeCoCmdBuilder() \
            .heuristic(HEURISTIC_F_MEASURE)
        self.run_cmd(builder, 'heuristic_f-measure')

    def test_heuristic_m_estimate(self):
        """
        Tests the SeCo algorithm when using the m-estimate heuristic for learning rules.
        """
        builder = SeCoCmdBuilder() \
            .heuristic(HEURISTIC_M_ESTIMATE)
        self.run_cmd(builder, 'heuristic_m-estimate')

    def test_pruning_heuristic_accuracy(self):
        """
        Tests the SeCo algorithm when using the accuracy heuristic for pruning rules.
        """
        builder = SeCoCmdBuilder() \
            .pruning(PRUNING_IREP) \
            .pruning_heuristic(HEURISTIC_ACCURACY)
        self.run_cmd(builder, 'pruning-heuristic_accuracy')

    def test_pruning_heuristic_precision(self):
        """
        Tests the SeCo algorithm when using the precision heuristic for pruning rules.
        """
        builder = SeCoCmdBuilder() \
            .pruning(PRUNING_IREP) \
            .pruning_heuristic(HEURISTIC_PRECISION)
        self.run_cmd(builder, 'pruning-heuristic_precision')

    def test_pruning_heuristic_recall(self):
        """
        Tests the SeCo algorithm when using the recall heuristic for pruning rules.
        """
        builder = SeCoCmdBuilder() \
            .pruning(PRUNING_IREP) \
            .pruning_heuristic(HEURISTIC_RECALL)
        self.run_cmd(builder, 'pruning-heuristic_recall')

    def test_pruning_heuristic_laplace(self):
        """
        Tests the SeCo algorithm when using the Laplace heuristic for pruning rules.
        """
        builder = SeCoCmdBuilder() \
            .pruning(PRUNING_IREP) \
            .pruning_heuristic(HEURISTIC_LAPLACE)
        self.run_cmd(builder, 'pruning-heuristic_laplace')

    def test_pruning_heuristic_wra(self):
        """
        Tests the SeCo algorithm when using the WRA heuristic for pruning rules.
        """
        builder = SeCoCmdBuilder() \
            .pruning(PRUNING_IREP) \
            .pruning_heuristic(HEURISTIC_WRA)
        self.run_cmd(builder, 'pruning-heuristic_wra')

    def test_pruning_heuristic_f_measure(self):
        """
        Tests the SeCo algorithm when using the F-measure heuristic for pruning rules.
        """
        builder = SeCoCmdBuilder() \
            .pruning(PRUNING_IREP) \
            .pruning_heuristic(HEURISTIC_F_MEASURE)
        self.run_cmd(builder, 'pruning-heuristic_f-measure')

    def test_pruning_heuristic_m_estimate(self):
        """
        Tests the SeCo algorithm when using the m-estimate heuristic for pruning rules.
        """
        builder = SeCoCmdBuilder() \
            .pruning(PRUNING_IREP) \
            .pruning_heuristic(HEURISTIC_M_ESTIMATE)
        self.run_cmd(builder, 'pruning-heuristic_m-estimate')
