"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from test_common import CommonIntegrationTests, CmdBuilder, DIR_DATA, DATASET_EMOTIONS

CMD_BOOMER = 'boomer'

FEATURE_BINNING_EQUAL_WIDTH = 'equal-width'

FEATURE_BINNING_EQUAL_FREQUENCY = 'equal-frequency'

LOSS_LOGISTIC_LABEL_WISE = 'logistic-label-wise'

LOSS_LOGISTIC_EXAMPLE_WISE = 'logistic-example-wise'

LOSS_SQUARED_HINGE_LABEL_WISE = 'squared-hinge-label-wise'

LOSS_SQUARED_ERROR_LABEL_WISE = 'squared-error-label-wise'

LOSS_SQUARED_ERROR_EXAMPLE_WISE = 'squared-error-example-wise'

HEAD_TYPE_SINGLE_LABEL = 'single-label'

HEAD_TYPE_COMPLETE = 'complete'

HEAD_TYPE_PARTIAL_FIXED = 'partial-fixed'

HEAD_TYPE_PARTIAL_DYNAMIC = 'partial-dynamic'

LABEL_BINNING_EQUAL_WIDTH = 'equal-width'

CLASSIFICATION_PREDICTOR_AUTO = 'auto'

CLASSIFICATION_PREDICTOR_LABEL_WISE = 'label-wise'

CLASSIFICATION_PREDICTOR_EXAMPLE_WISE = 'example-wise'

CLASSIFICATION_PREDICTOR_GFM = 'gfm'

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

    def classification_predictor(self, classification_predictor: str = CLASSIFICATION_PREDICTOR_AUTO):
        """
        Configures the algorithm to use a specific method for predicting binary labels.

        :param classification_predictor:    The name of the method that should be used for predicting binary labels
        :return:                            The builder itself
        """
        self.args.append('--classification-predictor')
        self.args.append(classification_predictor)
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

    def default_rule(self, default_rule: bool = True):
        """
        Configures whether the algorithm should induce a default rule or not.

        :param default_rule:    True, if a default rule should be induced, False otherwise
        :return:                The builder itself
        """
        self.args.append('--default-rule')
        self.args.append(str(default_rule).lower())
        return self

    def head_type(self, head_type: str = HEAD_TYPE_SINGLE_LABEL):
        """
        Configures the algorithm to use a specific type of rule heads.

        :param head_type:   The type of rule heads to be used
        :return:            The builder itself
        """
        self.args.append('--head-type')
        self.args.append(head_type)
        return self

    def label_binning(self, label_binning: str = LABEL_BINNING_EQUAL_WIDTH):
        """
        Configures the algorithm to use a specific method for the assignment of labels to bins.

        :param label_binning:   The name of the method to be used
        :return:                The builder itself
        """
        self.args.append('--label-binning')
        self.args.append(label_binning)
        return self

    def sparse_statistic_format(self, sparse: bool = True):
        """
        Configures whether sparse data structures should be used to store the statistics or not.

        :param sparse:  True, if sparse data structures should be used to store the statistics, False otherwise
        :return:        The builder itself
        """
        self.args.append('--statistic-format')
        self.args.append('sparse' if sparse else 'dense')
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

    def test_loss_squared_error_example_wise(self):
        """
        Tests the BOOMER algorithm when using the example-wise squared error loss function.
        """
        builder = BoostingCmdBuilder() \
            .loss(LOSS_SQUARED_ERROR_EXAMPLE_WISE)
        self.run_cmd(builder, self.cmd + '_loss-squared-error-example-wise')

    def test_predictor_classification_label_wise(self):
        """
        Tests the BOOMER algorithm when predicting binary labels that are obtained for each label individually.
        """
        builder = BoostingCmdBuilder() \
            .classification_predictor(CLASSIFICATION_PREDICTOR_LABEL_WISE) \
            .print_predictions(True)
        self.run_cmd(builder, self.cmd + '_predictor-classification-label-wise')

    def test_predictor_classification_example_wise(self):
        """
        Tests the BOOMER algorithm when predicting binary labels that are obtained by predicting one of the known label
        vectors.
        """
        builder = BoostingCmdBuilder() \
            .classification_predictor(CLASSIFICATION_PREDICTOR_EXAMPLE_WISE) \
            .print_predictions(True)
        self.run_cmd(builder, self.cmd + '_predictor-classification-example-wise')

    def test_predictor_classification_gfm(self):
        """
        Tests the BOOMER algorithm when predicting binary labels that are obtained via the general F-measure maximizer
        (GFM).
        """
        builder = BoostingCmdBuilder() \
            .classification_predictor(CLASSIFICATION_PREDICTOR_GFM) \
            .print_predictions(True)
        self.run_cmd(builder, self.cmd + '_predictor-classification-gfm')

    def test_predictor_probability_label_wise(self):
        """
        Tests the BOOMER algorithm when predicting probabilities that are obtained by applying a label-wise
        transformation function.
        """
        builder = BoostingCmdBuilder() \
            .predict_probabilities(True) \
            .probability_predictor(PROBABILITY_PREDICTOR_LABEL_WISE) \
            .print_predictions(True)
        self.run_cmd(builder, self.cmd + '_predictor-probability-label-wise')

    def test_predictor_probability_marginalized(self):
        """
        Tests the BOOMER algorithm when predicting probabilities that are obtained via marginalization over the known
        label vectors.
        """
        builder = BoostingCmdBuilder() \
            .predict_probabilities(True) \
            .probability_predictor(PROBABILITY_PREDICTOR_MARGINALIZED) \
            .print_predictions(True)
        self.run_cmd(builder, self.cmd + '_predictor-probability-marginalized')

    def test_no_default_rule(self):
        """
        Tests the BOOMER algorithm when not inducing a default rule.
        """
        builder = BoostingCmdBuilder() \
            .default_rule(False) \
            .print_model_characteristics(True)
        self.run_cmd(builder, self.cmd + '_no-default-rule')

    def test_statistics_sparse_labels_dense(self):
        """
        Tests the BOOMER algorithm when using sparse data structures for storing the statistics and a dense label
        representation.
        """
        builder = BoostingCmdBuilder(dataset=self.dataset_numerical) \
            .sparse_statistic_format(True) \
            .sparse_label_format(False) \
            .default_rule(False) \
            .loss(LOSS_SQUARED_HINGE_LABEL_WISE) \
            .head_type(HEAD_TYPE_SINGLE_LABEL)
        self.run_cmd(builder, self.cmd + '_statistics-sparse_labels-dense')

    def test_statistics_sparse_labels_sparse(self):
        """
        Tests the BOOMER algorithm when using sparse data structures for storing the statistics and a sparse label
        representation.
        """
        builder = BoostingCmdBuilder(dataset=self.dataset_numerical) \
            .sparse_statistic_format(True) \
            .sparse_label_format(True) \
            .default_rule(False) \
            .loss(LOSS_SQUARED_HINGE_LABEL_WISE) \
            .head_type(HEAD_TYPE_SINGLE_LABEL)
        self.run_cmd(builder, self.cmd + '_statistics-sparse_labels-sparse')

    def test_label_wise_single_label_heads(self):
        """
        Tests the BOOMER algorithm when using a label-wise decomposable loss function for the induction of rules with
        single-label heads.
        """
        builder = BoostingCmdBuilder() \
            .loss(LOSS_LOGISTIC_LABEL_WISE) \
            .head_type(HEAD_TYPE_SINGLE_LABEL) \
            .print_model_characteristics(True)
        self.run_cmd(builder, self.cmd + '_label-wise-single-label-heads')

    def test_label_wise_complete_heads(self):
        """
        Tests the BOOMER algorithm when using a label-wise decomposable loss function for the induction of rules with
        complete heads.
        """
        builder = BoostingCmdBuilder() \
            .loss(LOSS_LOGISTIC_LABEL_WISE) \
            .head_type(HEAD_TYPE_COMPLETE) \
            .print_model_characteristics(True)
        self.run_cmd(builder, self.cmd + '_label-wise-complete-heads')

    def test_label_wise_complete_heads_equal_width_label_binning(self):
        """
        Tests the BOOMER algorithm when using a label-wise decomposable loss function and equal-width label binning for
        the induction of rules with complete heads.
        """
        builder = BoostingCmdBuilder() \
            .loss(LOSS_LOGISTIC_LABEL_WISE) \
            .head_type(HEAD_TYPE_COMPLETE) \
            .label_binning(LABEL_BINNING_EQUAL_WIDTH) \
            .print_model_characteristics(True)
        self.run_cmd(builder, self.cmd + '_label-wise-complete-heads_equal-width-label-binning')

    def test_label_wise_partial_fixed_heads(self):
        """
        Tests the BOOMER algorithm when using a label-wise decomposable loss function for the induction of rules that
        predict for a predefined number of labels
        """
        builder = BoostingCmdBuilder() \
            .loss(LOSS_LOGISTIC_LABEL_WISE) \
            .head_type(HEAD_TYPE_PARTIAL_FIXED) \
            .print_model_characteristics(True)
        self.run_cmd(builder, self.cmd + '_label-wise-partial-fixed-heads')

    def test_label_wise_partial_fixed_heads_equal_width_label_binning(self):
        """
        Tests the BOOMER algorithm when using a label-wise decomposable loss function and equal-width label binning for
        the induction of rules that predict for a predefined number of labels
        """
        builder = BoostingCmdBuilder() \
            .loss(LOSS_LOGISTIC_LABEL_WISE) \
            .head_type(HEAD_TYPE_PARTIAL_FIXED) \
            .label_binning(LABEL_BINNING_EQUAL_WIDTH) \
            .print_model_characteristics(True)
        self.run_cmd(builder, self.cmd + '_label-wise-partial-fixed-heads_equal-width-label-binning')

    def test_label_wise_partial_dynamic_heads(self):
        """
        Tests the BOOMER algorithm when using a label-wise decomposable loss function for the induction of rules that
        predict for a dynamically determined subset of the available labels.
        """
        builder = BoostingCmdBuilder() \
            .loss(LOSS_LOGISTIC_LABEL_WISE) \
            .head_type(HEAD_TYPE_PARTIAL_DYNAMIC) \
            .print_model_characteristics(True)
        self.run_cmd(builder, self.cmd + '_label-wise-partial-dynamic-heads')

    def test_label_wise_partial_dynamic_heads_equal_width_label_binning(self):
        """
        Tests the BOOMER algorithm when using a label-wise decomposable loss function and equal-width label binning for
        the induction of rules that predict for a dynamically determined subset of the available labels.
        """
        builder = BoostingCmdBuilder() \
            .loss(LOSS_LOGISTIC_LABEL_WISE) \
            .head_type(HEAD_TYPE_PARTIAL_DYNAMIC) \
            .label_binning(LABEL_BINNING_EQUAL_WIDTH) \
            .print_model_characteristics(True)
        self.run_cmd(builder, self.cmd + '_label-wise-partial-dynamic-heads_equal-width-label-binning')

    def test_example_wise_single_label_heads(self):
        """
        Tests the BOOMER algorithm when using a non-decomposable loss function for the induction of rules with
        single-label heads.
        """
        builder = BoostingCmdBuilder() \
            .loss(LOSS_LOGISTIC_EXAMPLE_WISE) \
            .head_type(HEAD_TYPE_SINGLE_LABEL) \
            .print_model_characteristics(True)
        self.run_cmd(builder, self.cmd + '_example-wise-single-label-heads')

    def test_example_wise_complete_heads(self):
        """
        Tests the BOOMER algorithm when using a non-decomposable loss function for the induction of rules with complete
        heads.
        """
        builder = BoostingCmdBuilder() \
            .loss(LOSS_LOGISTIC_EXAMPLE_WISE) \
            .head_type(HEAD_TYPE_COMPLETE) \
            .print_model_characteristics(True)
        self.run_cmd(builder, self.cmd + '_example-wise-complete-heads')

    def test_example_wise_complete_heads_equal_width_label_binning(self):
        """
        Tests the BOOMER algorithm when using a non-decomposable loss function and equal-width label binning for the
        induction of rules with complete heads.
        """
        builder = BoostingCmdBuilder() \
            .loss(LOSS_LOGISTIC_EXAMPLE_WISE) \
            .head_type(HEAD_TYPE_COMPLETE) \
            .label_binning(LABEL_BINNING_EQUAL_WIDTH) \
            .print_model_characteristics(True)
        self.run_cmd(builder, self.cmd + '_example-wise-complete-heads_equal-width-label-binning')

    def test_example_wise_partial_fixed_heads(self):
        """
        Tests the BOOMER algorithm when using a non-decomposable loss function for the induction of rules that predict
        for a predefined number of labels
        """
        builder = BoostingCmdBuilder() \
            .loss(LOSS_LOGISTIC_EXAMPLE_WISE) \
            .head_type(HEAD_TYPE_PARTIAL_FIXED) \
            .print_model_characteristics(True)
        self.run_cmd(builder, self.cmd + '_example-wise-partial-fixed-heads')

    def test_example_wise_partial_fixed_heads_equal_width_label_binning(self):
        """
        Tests the BOOMER algorithm when using a non-decomposable loss function and equal-width label binning for the
        induction of rules that predict for a predefined number of labels
        """
        builder = BoostingCmdBuilder() \
            .loss(LOSS_LOGISTIC_EXAMPLE_WISE) \
            .head_type(HEAD_TYPE_PARTIAL_FIXED) \
            .label_binning(LABEL_BINNING_EQUAL_WIDTH) \
            .print_model_characteristics(True)
        self.run_cmd(builder, self.cmd + '_example-wise-partial-fixed-heads_equal-width-label-binning')

    def test_example_wise_partial_dynamic_heads(self):
        """
        Tests the BOOMER algorithm when using a non-decomposable loss function for the induction of rules that predict
        for a dynamically determined subset of the available labels.
        """
        builder = BoostingCmdBuilder() \
            .loss(LOSS_LOGISTIC_EXAMPLE_WISE) \
            .head_type(HEAD_TYPE_PARTIAL_DYNAMIC) \
            .print_model_characteristics(True)
        self.run_cmd(builder, self.cmd + '_example-wise-partial-dynamic-heads')

    def test_example_wise_partial_dynamic_heads_equal_width_label_binning(self):
        """
        Tests the BOOMER algorithm when using a non-decomposable loss function and equal-width label binning for the
        induction of rules that predict for a dynamically determined subset of the available labels.
        """
        builder = BoostingCmdBuilder() \
            .loss(LOSS_LOGISTIC_EXAMPLE_WISE) \
            .head_type(HEAD_TYPE_PARTIAL_DYNAMIC) \
            .label_binning(LABEL_BINNING_EQUAL_WIDTH) \
            .print_model_characteristics(True)
        self.run_cmd(builder, self.cmd + '_example-wise-partial-dynamic-heads_equal-width-label-binning')
