from mlrl.common.cython.learner cimport IRuleLearner, IRuleLearnerConfig, RuleLearner, RuleLearnerConfig

from libcpp.memory cimport unique_ptr


cdef extern from "seco/learner.hpp" namespace "seco" nogil:

    cdef cppclass ISeCoRuleLearnerConfig"seco::ISeCoRuleLearner::IConfig"(IRuleLearnerConfig):
        pass


    cdef cppclass ISeCoRuleLearner(IRuleLearner):
        pass


    unique_ptr[ISeCoRuleLearnerConfig] createSeCoRuleLearnerConfig()


    unique_ptr[ISeCoRuleLearner] createSeCoRuleLearner(unique_ptr[ISeCoRuleLearnerConfig] configPtr)


cdef class SeCoRuleLearnerConfig(RuleLearnerConfig):

    # Attributes:

    cdef unique_ptr[ISeCoRuleLearnerConfig] rule_learner_config_ptr


cdef class SeCoRuleLearner(RuleLearner):

    # Attributes:

    cdef unique_ptr[ISeCoRuleLearner] rule_learner_ptr
