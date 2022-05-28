"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from integration_tests import CmdBuilder, DIR_DATA, DATASET_EMOTIONS
from test_common import CommonIntegrationTests

CMD_BOOMER = 'boomer'

FEATURE_BINNING_EQUAL_WIDTH = 'equal-width'

FEATURE_BINNING_EQUAL_FREQUENCY = 'equal-frequency'


class BoostingCmdBuilder(CmdBuilder):
    """
    A builder that allows to configure a command for running the BOOMER algorithm.
    """

    def __init__(self, data_dir: str = DIR_DATA, dataset: str = DATASET_EMOTIONS):
        super(BoostingCmdBuilder, self).__init__(cmd=CMD_BOOMER, data_dir=data_dir, dataset=dataset)

    def feature_binning(self, feature_binning: str = FEATURE_BINNING_EQUAL_WIDTH):
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
        super(BoostingIntegrationTests, self).__init__(cmd=CMD_BOOMER, methodName=methodName)

    def test_feature_binning_equal_width_nominal_features_dense(self):
        """
        Tests the BOOMER algorithm's ability to use equal-width feature binning when applied to a dataset with nominal
        features using a dense feature representation.
        """
        builder = BoostingCmdBuilder(dataset=self.dataset_nominal) \
            .feature_binning(FEATURE_BINNING_EQUAL_WIDTH) \
            .sparse_feature_format(False)
        self.run_cmd(builder, self.cmd + '_feature-binning-equal-width_nominal-features-dense')

    def test_feature_binning_equal_width_nominal_features_sparse(self):
        """
        Tests the BOOMER algorithm's ability to use equal-width feature binning when applied to a dataset with nominal
        features using a sparse feature representation.
        """
        builder = BoostingCmdBuilder(dataset=self.dataset_nominal) \
            .feature_binning(FEATURE_BINNING_EQUAL_WIDTH) \
            .sparse_feature_format(True)
        self.run_cmd(builder, self.cmd + '_feature-binning-equal-width_nominal-features-sparse')

    def test_feature_binning_equal_width_numerical_features_dense(self):
        """
        Tests the BOOMER algorithm's ability to use equal-width feature binning when applied to a dataset with numerical
        features using a dense feature representation.
        """
        builder = BoostingCmdBuilder() \
            .feature_binning(FEATURE_BINNING_EQUAL_WIDTH) \
            .sparse_feature_format(False)
        self.run_cmd(builder, self.cmd + '_feature-binning-equal-width_numerical-features-dense')

    def test_feature_binning_equal_width_numerical_features_sparse(self):
        """
        Tests the BOOMER algorithm's ability to use equal-width feature binning when applied to a dataset with numerical
        features using a sparse feature representation.
        """
        builder = BoostingCmdBuilder() \
            .feature_binning(FEATURE_BINNING_EQUAL_WIDTH) \
            .sparse_feature_format(True)
        self.run_cmd(builder, self.cmd + '_feature-binning-equal-width_numerical-features-sparse')

    def test_feature_binning_equal_frequency_nominal_features_dense(self):
        """
        Tests the BOOMER algorithm's ability to use equal-frequency feature binning when applied to a dataset with
        nominal features using a dense feature representation.
        """
        builder = BoostingCmdBuilder(dataset=self.dataset_nominal) \
            .feature_binning(FEATURE_BINNING_EQUAL_FREQUENCY) \
            .sparse_feature_format(False)
        self.run_cmd(builder, self.cmd + '_feature-binning-equal-frequency_nominal-features-dense')

    def test_feature_binning_equal_frequency_nominal_features_sparse(self):
        """
        Tests the BOOMER algorithm's ability to use equal-frequency feature binning when applied to a dataset with
        nominal features using a sparse feature representation.
        """
        builder = BoostingCmdBuilder(dataset=self.dataset_nominal) \
            .feature_binning(FEATURE_BINNING_EQUAL_FREQUENCY) \
            .sparse_feature_format(True)
        self.run_cmd(builder, self.cmd + '_feature-binning-equal-frequency_nominal-features-sparse')

    def test_feature_binning_equal_frequency_numerical_features_dense(self):
        """
        Tests the BOOMER algorithm's ability to use equal-frequency feature binning when applied to a dataset with
        numerical features using a dense feature representation.
        """
        builder = BoostingCmdBuilder() \
            .feature_binning(FEATURE_BINNING_EQUAL_FREQUENCY) \
            .sparse_feature_format(False)
        self.run_cmd(builder, self.cmd + '_feature-binning-equal-frequency_numerical-features-dense')

    def test_feature_binning_equal_frequency_numerical_features_sparse(self):
        """
        Tests the BOOMER algorithm's ability to use equal-frequency feature binning when applied to a dataset with
        numerical features using a sparse feature representation.
        """
        builder = BoostingCmdBuilder() \
            .feature_binning(FEATURE_BINNING_EQUAL_FREQUENCY) \
            .sparse_feature_format(True)
        self.run_cmd(builder, self.cmd + '_feature-binning-equal-frequency_numerical-features-sparse')
