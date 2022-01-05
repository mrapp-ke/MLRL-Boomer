from mlrl.common.cython.learner cimport AbstractRuleLearner, RuleLearner


cdef extern from "boosting/learner.hpp" namespace "boosting" nogil:

    cdef cppclass BoostingRuleLearnerImpl"boosting::BoostingRuleLearner"(AbstractRuleLearner):
        pass


cdef class BoostingRuleLearner(RuleLearner):
    pass
