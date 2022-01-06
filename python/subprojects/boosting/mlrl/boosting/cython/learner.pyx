"""
@author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from libcpp.memory cimport make_unique


cdef class BoostingRuleLearner(RuleLearner):
    """
    A wrapper for the C++ class `BoostingRuleLearner`.
    """

    def __cinit__(self):
        self.rule_learner_ptr = make_unique[BoostingRuleLearnerImpl](make_unique[BoostingRuleLearnerConfigImpl]())

    cdef IRuleLearner* get_rule_learner_ptr(self):
        return self.rule_learner_ptr.get()
