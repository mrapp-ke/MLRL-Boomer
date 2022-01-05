"""
@author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from libcpp.memory cimport unique_ptr, make_unique


cdef class BoostingRuleLearner(RuleLearner):
    """
    A wrapper for the C++ class `BoostingRuleLearner`.
    """

    def __cinit__(self):
        self.rule_learner_ptr = <unique_ptr[AbstractRuleLearner]>make_unique[BoostingRuleLearnerImpl]()
