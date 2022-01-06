"""
@author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from libcpp.memory cimport make_unique


cdef class SeCoRuleLearner(RuleLearner):
    """
    A wrapper for the C++ class `SeCoRuleLearner`.
    """

    def __cinit__(self):
        self.rule_learner_ptr = make_unique[SeCoRuleLearnerImpl](SeCoRuleLearnerConfigImpl())

    cdef AbstractRuleLearner* get_rule_learner_ptr(self):
        return self.rule_learner_ptr.get()
