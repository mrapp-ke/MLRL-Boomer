"""
@author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from libcpp.memory cimport make_unique
from libcpp.utility cimport move


cdef class SeCoRuleLearnerConfig(RuleLearnerConfig):
    """
    A wrapper for the pure virtual C++ class `ISeCoRuleLearner::IConfig`.
    """

    def __cinit__(self):
        self.rule_learner_config_ptr = createSeCoRuleLearnerConfig()

    cdef IRuleLearnerConfig* get_rule_learner_config_ptr(self):
        return self.rule_learner_config_ptr.get()


cdef class SeCoRuleLearner(RuleLearner):
    """
    A wrapper for the pure virtual C++ class `ISeCoRuleLearner`.
    """

    def __cinit__(self, SeCoRuleLearnerConfig config not None):
        """
        :param config: The configuration that should be used by the rule learner
        """
        self.rule_learner_ptr = createSeCoRuleLearner(move(config.rule_learner_config_ptr))

    cdef IRuleLearner* get_rule_learner_ptr(self):
        return self.rule_learner_ptr.get()
