"""
@author: Michael Rapp (michael.rapp.ml@gmail.com)
"""


cdef class BoostingRuleLearnerConfig:
    """
    Allows to configure a rule learner that makes use of gradient boosting.
    """

    cdef IBoostingRuleLearnerConfig* get_boosting_rule_learner_config_ptr(self):
        pass

    def use_label_wise_probability_predictor(self):
        """
        Configures the rule learner to use a predictor for predicting probability estimates by summing up the scores
        that are provided by individual rules of an existing rule-based model and transforming the aggregated scores
        into probabilities according to a certain transformation function that is applied to each label individually.
        """
        cdef IBoostingRuleLearnerConfig* rule_learner_config_ptr = self.get_boosting_rule_learner_config_ptr()
        rule_learner_config_ptr.useLabelWiseProbabilityPredictor()
