from libcpp.memory cimport unique_ptr


cdef extern from "cpp/model/rule_model.h" nogil:

    cdef cppclass RuleModelImpl"RuleModel":
        pass


cdef extern from "cpp/model/model_builder.h" nogil:

    cdef cppclass IModelBuilder:
        pass


cdef class RuleModel:

    # Attributes:

    cdef unique_ptr[RuleModelImpl] model_ptr


cdef class ModelBuilder:

    # Attributes:

    cdef unique_ptr[IModelBuilder] model_builder_ptr
