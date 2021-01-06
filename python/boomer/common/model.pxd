from boomer.common.head_refinement cimport AbstractPrediction
from boomer.common.rules cimport ConditionList

from libcpp.memory cimport unique_ptr, shared_ptr


cdef extern from "cpp/model/rule_model.h" nogil:

    cdef cppclass RuleModelImpl"RuleModel":
        pass


cdef extern from "cpp/model/model_builder.h" nogil:

    cdef cppclass IModelBuilder:

        # Functions:

        void setDefaultRule(const AbstractPrediction& prediction)

        void addRule(const ConditionList& conditions, const AbstractPrediction& prediction)

        unique_ptr[RuleModelImpl] build()


cdef class RuleModel:

    # Attributes:

    cdef unique_ptr[RuleModelImpl] model_ptr


cdef class ModelBuilder:

    # Attributes:

    cdef shared_ptr[IModelBuilder] model_builder_ptr

    # Functions:

    cdef RuleModel build(self)
