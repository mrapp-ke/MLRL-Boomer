"""
@author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from mlrl.boosting.cython.prediction cimport LabelWiseBinaryPredictorConfig


cdef class BoostingRuleLearnerConfig(RuleLearnerConfig):
    """
    Allows to configure a rule learner that makes use of gradient boosting.
    """

    cdef IBoostingRuleLearnerConfig* get_boosting_rule_learner_config_ptr(self):
        pass

    def use_complete_heads(self):
        """
        Configures the rule learner to induce rules with complete heads that predict for all available labels.
        """
        cdef IBoostingRuleLearnerConfig* rule_learner_config_ptr = self.get_boosting_rule_learner_config_ptr()
        rule_learner_config_ptr.useCompleteHeads()

    def use_dense_statistics(self):
        """
        Configures the rule learner to use a dense representation of gradients and Hessians.
        """
        cdef IBoostingRuleLearnerConfig* rule_learner_config_ptr = self.get_boosting_rule_learner_config_ptr()
        rule_learner_config_ptr.useDenseStatistics()

    def use_no_l1_regularization(self):
        """
        Configures the rule learner to not use L1 regularization.
        """
        cdef IBoostingRuleLearnerConfig* rule_learner_config_ptr = self.get_boosting_rule_learner_config_ptr()
        rule_learner_config_ptr.useNoL1Regularization()

    def use_no_l2_regularization(self):
        """
        Configures the rule learner to not use L2 regularization.
        """
        cdef IBoostingRuleLearnerConfig* rule_learner_config_ptr = self.get_boosting_rule_learner_config_ptr()
        rule_learner_config_ptr.useNoL2Regularization()

    def use_label_wise_logistic_loss(self):
        """
        Configures the rule learner to use a loss function that implements a multi-label variant of the logistic loss
        that is applied label-wise.
        """
        cdef IBoostingRuleLearnerConfig* rule_learner_config_ptr = self.get_boosting_rule_learner_config_ptr()
        rule_learner_config_ptr.useLabelWiseLogisticLoss()

    def use_no_label_binning(self):
        """
        Configures the rule learner to not use any method for the assignment of labels to bins.
        """
        cdef IBoostingRuleLearnerConfig* rule_learner_config_ptr = self.get_boosting_rule_learner_config_ptr()
        rule_learner_config_ptr.useNoLabelBinning()

    def use_label_wise_binary_predictor(self) -> LabelWiseBinaryPredictorConfig:
        """
        Configures the rule learner to use a predictor for predicting whether individual labels are relevant or
        irrelevant by summing up the scores that are provided by the individual rules of an existing rule-based model
        and transforming them into binary values according to a certain threshold that is applied to each label
        individually.

        :return: A `LabelWiseBinaryPredictorConfig` that allows further configuration of the predictor
        """
        cdef IBoostingRuleLearnerConfig* rule_learner_config_ptr = self.get_boosting_rule_learner_config_ptr()
        cdef ILabelWiseBinaryPredictorConfig* config_ptr = &rule_learner_config_ptr.useLabelWiseBinaryPredictor()
        cdef LabelWiseBinaryPredictorConfig config = LabelWiseBinaryPredictorConfig.__new__(LabelWiseBinaryPredictorConfig)
        config.config_ptr = config_ptr
        return config

    def use_label_wise_score_predictor(self):
        """
        Configures the rule learner to use a predictor for predicting regression scores by summing up the scores that
        are provided by the individual rules of an existing rule-based model for each label individually.
        """
        cdef IBoostingRuleLearnerConfig* rule_learner_config_ptr = self.get_boosting_rule_learner_config_ptr()
        rule_learner_config_ptr.useLabelWiseScorePredictor()

    def use_label_wise_probability_predictor(self):
        """
        Configures the rule learner to use a predictor for predicting probability estimates by summing up the scores
        that are provided by individual rules of an existing rule-based model and transforming the aggregated scores
        into probabilities according to a certain transformation function that is applied to each label individually.
        """
        cdef IBoostingRuleLearnerConfig* rule_learner_config_ptr = self.get_boosting_rule_learner_config_ptr()
        rule_learner_config_ptr.useLabelWiseProbabilityPredictor()
