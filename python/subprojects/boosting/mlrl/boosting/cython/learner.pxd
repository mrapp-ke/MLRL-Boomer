from mlrl.common.cython.learner cimport AbstractRuleLearner, RuleLearner, AbstractRuleLearnerConfigImpl

from libcpp.memory cimport unique_ptr


cdef extern from "boosting/learner.hpp" namespace "boosting" nogil:

    cdef cppclass BoostingRuleLearnerConfigImpl"boosting::BoostingRuleLearner::Config"(AbstractRuleLearnerConfigImpl):

        # Constructors:

        BoostingRuleLearnerConfigImpl()


    cdef cppclass BoostingRuleLearnerImpl"boosting::BoostingRuleLearner"(AbstractRuleLearner):

        # Constructors:

        BoostingRuleLearnerImpl(BoostingRuleLearnerConfigImpl config)


cdef class BoostingRuleLearner(RuleLearner):

    # Attributes:

    cdef unique_ptr[BoostingRuleLearnerImpl] rule_learner_ptr
