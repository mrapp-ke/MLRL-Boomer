from mlrl.common.cython.learner cimport RuleLearner, AbstractRuleLearner, AbstractRuleLearnerConfigImpl

from libcpp.memory cimport unique_ptr


cdef extern from "seco/learner.hpp" namespace "seco" nogil:

    cdef cppclass SeCoRuleLearnerConfigImpl"seco::SeCoRuleLearner::Config"(AbstractRuleLearnerConfigImpl):

        # Constructors:

        SeCoRuleLearnerConfigImpl()


    cdef cppclass SeCoRuleLearnerImpl"seco::SeCoRuleLearner"(AbstractRuleLearner):

        # Constructors:

        SeCoRuleLearner(SeCoRuleLearnerConfigImpl config)


cdef class SeCoRuleLearner(RuleLearner):

    # Attributes:

    cdef unique_ptr[SeCoRuleLearnerImpl] rule_learner_ptr
