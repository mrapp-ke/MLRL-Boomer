"""
@author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from libcpp.memory cimport unique_ptr, make_unique


cdef class SeCoRuleLearner(RuleLearner):
    """
    A wrapper for the C++ class `SeCoRuleLearner`.
    """

    def __cinit__(self):
        self.rule_learner_ptr = <unique_ptr[AbstractRuleLearner]>make_unique[SeCoRuleLearnerImpl]()
