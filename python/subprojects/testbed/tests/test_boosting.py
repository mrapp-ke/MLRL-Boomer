"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from integration_tests import CmdBuilder, DIR_DATA, DATASET_EMOTIONS
from test_common import CommonIntegrationTests

CMD_BOOMER = 'boomer'

FEATURE_BINNING_EQUAL_WIDTH = 'equal-width'

FEATURE_BINNING_EQUAL_FREQUENCY = 'equal-frequency'

LOSS_LOGISTIC_LABEL_WISE = 'logistic-label-wise'

LOSS_LOGISTIC_EXAMPLE_WISE = 'logistic-example-wise'

LOSS_SQUARED_HINGE_LABEL_WISE = 'squared-hinge-label-wise'

LOSS_SQUARED_ERROR_LABEL_WISE = 'squared-error-label-wise'

PROBABILITY_PREDICTOR_AUTO = 'auto'

PROBABILITY_PREDICTOR_LABEL_WISE = 'label-wise'

PROBABILITY_PREDICTOR_MARGINALIZED = 'marginalized'


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

    def loss(self, loss: str = LOSS_LOGISTIC_LABEL_WISE):
        """
        Configures the algorithm to use a specific loss function.

        :param loss:    The name of the loss function that should be used
        :return:        The builder itself
        """
        self.args.append('--loss')
        self.args.append(loss)
        return self

    def probability_predictor(self, probability_predictor: str = PROBABILITY_PREDICTOR_AUTO):
        """
        Configures the algorithm to use a specific method for predicting probabilities.

        :param probability_predictor:   The name of the method that should be used for predicting probabilities
        :return:                        The builder itself
        """
        self.args.append('--probability-predictor')
        self.args.append(probability_predictor)
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

    def test_loss_logistic_label_wise(self):
        """
        Tests the BOOMER algorithm when using the label-wise logistic loss function.
        """
        builder = BoostingCmdBuilder() \
            .loss(LOSS_LOGISTIC_LABEL_WISE)
        self.run_cmd(builder, self.cmd + '_loss-logistic-label-wise')

    def test_loss_logistic_example_wise(self):
        """
        Tests the BOOMER algorithm when using the example-wise logistic loss function.
        """
        builder = BoostingCmdBuilder() \
            .loss(LOSS_LOGISTIC_EXAMPLE_WISE)
        self.run_cmd(builder, self.cmd + '_loss-logistic-example-wise')

    def test_loss_squared_hinge_label_wise(self):
        """
        Tests the BOOMER algorithm when using the label-wise squared hinge loss function.
        """
        builder = BoostingCmdBuilder() \
            .loss(LOSS_SQUARED_HINGE_LABEL_WISE)
        self.run_cmd(builder, self.cmd + '_loss-squared-hinge-label-wise')

    def test_loss_squared_error_label_wise(self):
        """
        Tests the BOOMER algorithm when using the label-wise squared error loss function.
        """
        builder = BoostingCmdBuilder() \
            .loss(LOSS_SQUARED_ERROR_LABEL_WISE)
        self.run_cmd(builder, self.cmd + '_loss-squared-error-label-wise')

    def test_probabilities_label_wise(self):
        """
        Tests the BOOMER algorithm when predicting probabilities that are obtained by applying a label-wise
        transformation function.
        """
        builder = BoostingCmdBuilder() \
            .predict_probabilities(True) \
            .probability_predictor(PROBABILITY_PREDICTOR_LABEL_WISE) \
            .print_predictions(True)
        self.run_cmd(builder, self.cmd + '_predictor-probability-label-wise')

    def test_probabilities_marginalized(self):
        """
        Tests the BOOMER algorithm when predicting probabilities that are obtained via marginalization over the known
        label vectors.
        """
        builder = BoostingCmdBuilder() \
            .predict_probabilities(True) \
            .probability_predictor(PROBABILITY_PREDICTOR_MARGINALIZED) \
            .print_predictions(True)
        self.run_cmd(builder, self.cmd + '_predictor-probability-marginalized')
