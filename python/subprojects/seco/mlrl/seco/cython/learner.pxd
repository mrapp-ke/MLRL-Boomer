from mlrl.common.cython.learner cimport RuleLearner, AbstractRuleLearner


cdef extern from "seco/learner.hpp" namespace "seco" nogil:

    cdef cppclass SeCoRuleLearnerImpl"seco::SeCoRuleLearner"(AbstractRuleLearner):
        pass


cdef class SeCoRuleLearner(RuleLearner):
    pass
