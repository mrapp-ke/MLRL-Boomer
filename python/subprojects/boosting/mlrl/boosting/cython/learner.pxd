from mlrl.common.cython.learner cimport IRuleLearner, IRuleLearnerConfig, RuleLearner

from libcpp.memory cimport unique_ptr


cdef extern from "boosting/learner.hpp" namespace "boosting" nogil:

    cdef cppclass BoostingRuleLearnerConfigImpl"boosting::BoostingRuleLearner::Config"(IRuleLearnerConfig):

        # Constructors:

        BoostingRuleLearnerConfigImpl()


    cdef cppclass BoostingRuleLearnerImpl"boosting::BoostingRuleLearner"(IRuleLearner):

        # Constructors:

        BoostingRuleLearnerImpl(unique_ptr[BoostingRuleLearnerConfigImpl] configPtr)


cdef class BoostingRuleLearner(RuleLearner):

    # Attributes:

    cdef unique_ptr[BoostingRuleLearnerImpl] rule_learner_ptr
