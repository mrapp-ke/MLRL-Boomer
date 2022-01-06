from mlrl.common.cython.learner cimport IRuleLearner, IRuleLearnerConfig, RuleLearner

from libcpp.memory cimport unique_ptr


cdef extern from "seco/learner.hpp" namespace "seco" nogil:

    cdef cppclass SeCoRuleLearnerConfigImpl"seco::SeCoRuleLearner::Config"(IRuleLearnerConfig):

        # Constructors:

        SeCoRuleLearnerConfigImpl()


    cdef cppclass SeCoRuleLearnerImpl"seco::SeCoRuleLearner"(IRuleLearner):

        # Constructors:

        SeCoRuleLearner(unique_ptr[SeCoRuleLearnerConfigImpl] configPtr)


cdef class SeCoRuleLearner(RuleLearner):

    # Attributes:

    cdef unique_ptr[SeCoRuleLearnerImpl] rule_learner_ptr
