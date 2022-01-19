"""
@author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from mlrl.boosting.cython.label_binning cimport EqualWidthLabelBinningConfig
from mlrl.boosting.cython.loss cimport ExampleWiseLogisticLossConfig, LabelWiseLogisticLossConfig, \
    LabelWiseSquaredErrorLossConfig, LabelWiseSquaredHingeLossConfig
from mlrl.boosting.cython.post_processor cimport ConstantShrinkageConfig
from mlrl.boosting.cython.predictor cimport ExampleWiseClassificationPredictorConfig, \
    LabelWiseClassificationPredictorConfig, LabelWiseRegressionPredictorConfig, LabelWiseProbabilityPredictorConfig

from libcpp.memory cimport make_unique
from libcpp.utility cimport move


cdef class BoostingRuleLearnerConfig(RuleLearnerConfig):
    """
    A wrapper for the pure virtual C++ class `IBoostingRuleLearner::IConfig`.
    """

    def __cinit__(self):
        self.rule_learner_config_ptr = createBoostingRuleLearnerConfig()

    cdef IRuleLearnerConfig* get_rule_learner_config_ptr(self):
        return self.rule_learner_config_ptr.get()

    def use_constant_shrinkage_post_processor(self) -> ConstantShrinkageConfig:
        """
        Configures the rule learner to use a post-processor that shrinks the weights of rules by a constant "shrinkage"
        parameter.

        :return: A `ConstantShrinkageConfig` that allows further configuration of the post-processor
        """
        cdef IBoostingRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        cdef IConstantShrinkageConfig* config_ptr = &rule_learner_config_ptr.useConstantShrinkagePostProcessor()
        cdef ConstantShrinkageConfig config = ConstantShrinkageConfig.__new__(ConstantShrinkageConfig)
        config.config_ptr = config_ptr
        return config

    def use_example_wise_logistic_loss(self) -> ExampleWiseLogisticLossConfig:
        """
        Configures the rule learner to use a loss function that implements a multi-label variant of the logistic loss
        that is applied example-wise.

        :return: An `ExampleWiseLogisticLossConfig` that allows further configuration of the loss function
        """
        cdef IBoostingRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        cdef ExampleWiseLogisticLossConfigImpl* config_ptr = &rule_learner_config_ptr.useExampleWiseLogisticLoss()
        cdef ExampleWiseLogisticLossConfig config = ExampleWiseLogisticLossConfig.__new__(ExampleWiseLogisticLossConfig)
        config.config_ptr = config_ptr
        return config

    def use_label_wise_logistic_loss(self) -> LabelWiseLogisticLossConfig:
        """
        Configures the rule learner to use a loss function that implements a multi-label variant of the logistic loss
        that is applied label-wise.

        :return: A `LabelWiseLogisticLossConfig` that allows further configuration of the loss function
        """
        cdef IBoostingRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        cdef LabelWiseLogisticLossConfigImpl* config_ptr = &rule_learner_config_ptr.useLabelWiseLogisticLoss()
        cdef LabelWiseLogisticLossConfig config = LabelWiseLogisticLossConfig.__new__(LabelWiseLogisticLossConfig)
        config.config_ptr = config_ptr
        return config

    def use_label_wise_squared_error_loss(self) -> LabelWiseSquaredErrorLossConfig:
        """
        Configures the rule learner to use a loss function that implements a multi-label variant of the squared error
        loss that is applied label-wise.

        :return: A `LabelWiseSquaredErrorLossConfig` that allows further configuration of the loss function
        """
        cdef IBoostingRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        cdef LabelWiseSquaredErrorLossConfigImpl* config_ptr = &rule_learner_config_ptr.useLabelWiseSquaredErrorLoss()
        cdef LabelWiseSquaredErrorLossConfig config = LabelWiseSquaredErrorLossConfig.__new__(LabelWiseSquaredErrorLossConfig)
        config.config_ptr = config_ptr
        return config

    def use_label_wise_squared_hinge_loss(self) -> LabelWiseSquaredHingeLossConfig:
        """
        Configures the rule learner to use a loss function that implements a multi-label variant of the squared hinge
        loss that is applied label-wise.

        :return: A `LabelWiseSquaredHingeLossConfig` that allows further configuration of the loss function
        """
        cdef IBoostingRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        cdef LabelWiseSquaredHingeLossConfigImpl* config_ptr = &rule_learner_config_ptr.useLabelWiseSquaredHingeLoss()
        cdef LabelWiseSquaredHingeLossConfig config = LabelWiseSquaredHingeLossConfig.__new__(LabelWiseSquaredHingeLossConfig)
        config.config_ptr = config_ptr
        return config

    def use_no_label_binning(self):
        """
        Configures the algorithm to not use any method for the assignment of labels to bins.
        """
        cdef IBoostingRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        rule_learner_config_ptr.useNoLabelBinning()

    def use_equal_width_label_binning(self) -> EqualWidthLabelBinningConfig:
        """
        Configures the algorithm to use a method for the assignment of labels to bins in a way such that each bin
        contains labels for which the predicted score is expected to belong to the same value range.

        :return: A `EqualWidthLabelBinningConfig` that allows further configuration of the method for the assignment of
                 labels to bins
        """
        cdef IBoostingRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        cdef EqualWidthLabelBinningConfigImpl* config_ptr = &rule_learner_config_ptr.useEqualWidthLabelBinning()
        cdef EqualWidthLabelBinningConfig config = EqualWidthLabelBinningConfig.__new__(EqualWidthLabelBinningConfig)
        config.config_ptr = config_ptr
        return config

    def use_example_wise_classification_predictor(self) -> ExampleWiseClassificationPredictorConfig:
        """
        Configures the algorithm to use a predictor for predicting whether individual labels are relevant or irrelevant
        by summing up the scores that are provided by an existing rule-based model and comparing the aggregated score
        vector to the known label vectors according to a certain distance measure. The label vector that is closest to
        the aggregated score vector is finally predicted.

        :return: An `ExampleWiseClassificationPredictorConfig` that allows further configuration of the predictor for
                 predicting whether individual labels are relevant or irrelevant
        """
        cdef IBoostingRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        cdef IExampleWiseClassificationPredictorConfig* config_ptr = &rule_learner_config_ptr.useExampleWiseClassificationPredictor()
        cdef ExampleWiseClassificationPredictorConfig config = ExampleWiseClassificationPredictorConfig.__new__(ExampleWiseClassificationPredictorConfig)
        config.config_ptr = config_ptr
        return config

    def use_label_wise_classification_predictor(self) -> LabelWiseClassificationPredictorConfig:
        """
        Configures the algorithm to use a predictor for predicting whether individual labels are relevant or irrelevant
        by summing up the scores that are provided by the individual rules of an existing rule-based model and
        transforming them into binary values according to a certain threshold that is applied to each label
        individually.

        :return: A `LabelWiseClassificationPredictorConfig` that allows further configuration of the predictor for
                 predicting whether individual labels are relevant or irrelevant
        """
        cdef IBoostingRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        cdef ILabelWiseClassificationPredictorConfig* config_ptr = &rule_learner_config_ptr.useLabelWiseClassificationPredictor()
        cdef LabelWiseClassificationPredictorConfig config = LabelWiseClassificationPredictorConfig.__new__(LabelWiseClassificationPredictorConfig)
        config.config_ptr = config_ptr
        return config

    def use_label_wise_regression_predictor(self) -> LabelWiseRegressionPredictorConfig:
        """
        Configures the algorithm to use a predictor for predicting regression scores by summing up the scores that are
        provided by the individual rules of an existing rule-based model for each label individually.

        :return: A `LabelWiseRegressionPredictorConfig` that allows further configuration of the predictor for
                 predicting regression scores
        """
        cdef IBoostingRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        cdef ILabelWiseRegressionPredictorConfig* config_ptr = &rule_learner_config_ptr.useLabelWiseRegressionPredictor()
        cdef LabelWiseRegressionPredictorConfig config = LabelWiseRegressionPredictorConfig.__new__(LabelWiseRegressionPredictorConfig)
        config.config_ptr = config_ptr
        return config

    def use_label_wise_probability_predictor(self) -> LabelWiseProbabilityPredictorConfig:
        """
        Configures the algorithm to use a predictor for predicting probability estimates by summing up the scores that
        are provided by individual rules of an existing rule-based models and transforming the aggregated scores into
        probabilities according to a certain transformation function that is applied to each label individually.

        :return: A `LabelWiseProbabilityPredictorConfig` that allows further configuration of the predictor for
                 predicting probability estimates
        """
        cdef IBoostingRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        cdef ILabelWiseProbabilityPredictorConfig* config_ptr = &rule_learner_config_ptr.useLabelWiseProbabilityPredictor()
        cdef LabelWiseProbabilityPredictorConfig config = LabelWiseProbabilityPredictorConfig.__new__(LabelWiseProbabilityPredictorConfig)
        config.config_ptr = config_ptr
        return config


cdef class BoostingRuleLearner(RuleLearner):
    """
    A wrapper for the pure virtual C++ class `IBoostingRuleLearner`.
    """

    def __cinit__(self, BoostingRuleLearnerConfig config not None):
        """
        :param config: The configuration that should be used by the rule learner
        """
        self.rule_learner_ptr = createBoostingRuleLearner(move(config.rule_learner_config_ptr))

    cdef IRuleLearner* get_rule_learner_ptr(self):
        return self.rule_learner_ptr.get()
