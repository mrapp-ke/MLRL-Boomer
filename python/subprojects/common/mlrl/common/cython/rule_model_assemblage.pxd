from libcpp cimport bool


cdef extern from "common/rule_induction/rule_model_assemblage_sequential.hpp" nogil:

    cdef cppclass SequentialRuleModelAssemblageConfigImpl"SequentialRuleModelAssemblageConfig":

        # Functions:

        bool getUseDefaultRule() const

        SequentialRuleModelAssemblageConfigImpl& setUseDefaultRule(bool useDefaultRule) except +


cdef class SequentialRuleModelAssemblageConfig:

    # Attributes:

    cdef SequentialRuleModelAssemblageConfigImpl* config_ptr
