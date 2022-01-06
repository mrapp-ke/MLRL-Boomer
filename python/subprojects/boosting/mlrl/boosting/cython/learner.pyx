"""
@author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
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
