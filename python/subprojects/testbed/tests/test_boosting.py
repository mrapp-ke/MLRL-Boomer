"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from integration_tests import CmdBuilder, DIR_DATA, DATASET_EMOTIONS
from test_common import CommonIntegrationTests

CMD_BOOMER = 'boomer'


class BoostingCmdBuilder(CmdBuilder):
    """
    A builder that allows to configure a command for running the BOOMER algorithm.
    """

    def __init__(self, data_dir: str = DIR_DATA, dataset: str = DATASET_EMOTIONS):
        super(BoostingCmdBuilder, self).__init__(CMD_BOOMER, data_dir, dataset)

    def feature_binning(self, feature_binning: str = 'equal-width'):
        """
        Configures the algorithm to use a specific method for feature binning.

        :param feature_binning: The name of the method that should be used for feature binning
        :return:                The builder itself
        """
        self.args.append('--feature-binning')
        self.args.append(feature_binning)
        return self


class BoostingIntegrationTests(CommonIntegrationTests):
    """
    Defines a series of integration tests for the BOOMER algorithm.
    """

    def __init__(self, methodName='runTest'):
        """
        :param methodName: The name of the test method to be executed
        """
        super(BoostingIntegrationTests, self).__init__(CMD_BOOMER, methodName=methodName)

    def test_feature_binning_equal_width_nominal_features_dense(self):
        """
        Tests the BOOMER algorithm's ability to use equal-width feature binning when applied to a dataset with nominal
        features that are represented by using dense data structures.
        """
        builder = BoostingCmdBuilder(self.cmd, dataset=self.dataset_nominal) \
            .feature_binning('equal-width')
        self.run_cmd(builder, self.cmd + '_feature-binning-equal-width_nominal-features-dense')
