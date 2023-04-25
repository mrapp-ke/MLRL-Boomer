"""
@author: Michael Rapp (michael.rapp.ml@gmail.com)
"""


cdef class SeCoRuleLearnerConfig:
    """
    Allows to configure a rule learner that makes use of the separate-and-conquer (SeCo) paradigm.
    """

    cdef ISeCoRuleLearnerConfig* get_seco_rule_learner_config_ptr(self):
        pass

    def use_no_coverage_stopping_criterion(self):
        """
        Configures the rule learner to not use any stopping criterion that stops the induction of rules as soon as the
        sum of the weights of the uncovered labels is smaller or equal to a certain threshold.
        """
        cdef ISeCoRuleLearnerConfig* rule_learner_config_ptr = self.get_seco_rule_learner_config_ptr()
        rule_learner_config_ptr.useNoCoverageStoppingCriterion()

    def use_single_label_heads(self):
        """
        Configures the rule learner to induce rules with single-label heads that predict for a single label.
        """
        cdef ISeCoRuleLearnerConfig* rule_learner_config_ptr = self.get_seco_rule_learner_config_ptr()
        rule_learner_config_ptr.useSingleLabelHeads()

    def use_precision_heuristic(self):
        """
        Configures the rule learner to use the "Precision" heuristic for learning rules.
        """
        cdef ISeCoRuleLearnerConfig* rule_learner_config_ptr = self.get_seco_rule_learner_config_ptr()
        rule_learner_config_ptr.usePrecisionHeuristic()

    def use_precision_pruning_heuristic(self):
        """
        Configures the rule learner to use the "Precision" heuristic for pruning rules.
        """
        cdef ISeCoRuleLearnerConfig* rule_learner_config_ptr = self.get_seco_rule_learner_config_ptr()
        rule_learner_config_ptr.usePrecisionPruningHeuristic()

    def use_label_wise_binary_predictor(self):
        """
        Configures the rule learner to use predictor for predicting whether individual labels of given query examples
        are relevant or irrelevant by processing rules of an existing rule-based model in the order they have been
        learned. If a rule covers an example, its prediction is applied to each label individually.
        """
        cdef ISeCoRuleLearnerConfig* rule_learner_config_ptr = self.get_seco_rule_learner_config_ptr()
        rule_learner_config_ptr.useLabelWiseBinaryPredictor()
