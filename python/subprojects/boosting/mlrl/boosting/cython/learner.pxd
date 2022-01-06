from mlrl.common.cython.learner cimport IRuleLearner, IRuleLearnerConfig, RuleLearner, RuleLearnerConfig

from libcpp.memory cimport unique_ptr


cdef extern from "boosting/learner.hpp" namespace "boosting" nogil:

    cdef cppclass IBoostingRuleLearnerConfig"boosting::IBoostingRuleLearner::IConfig"(IRuleLearnerConfig):
        pass


    cdef cppclass IBoostingRuleLearner(IRuleLearner):
        pass


    unique_ptr[IBoostingRuleLearnerConfig] createBoostingRuleLearnerConfig()


    unique_ptr[IBoostingRuleLearner] createBoostingRuleLearner(unique_ptr[IBoostingRuleLearnerConfig] configPtr)


cdef class BoostingRuleLearnerConfig(RuleLearnerConfig):

    # Attributes:

    cdef unique_ptr[IBoostingRuleLearnerConfig] rule_learner_config_ptr


cdef class BoostingRuleLearner(RuleLearner):

    # Attributes:

    cdef unique_ptr[IBoostingRuleLearner] rule_learner_ptr
