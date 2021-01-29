from boomer.common.model cimport IModelBuilder, ModelBuilder


cdef extern from "cpp/model/rule_list.h" namespace "boosting" nogil:

    cdef cppclass RuleListBuilderImpl"boosting::RuleListBuilder"(IModelBuilder):
        pass


cdef class RuleListBuilder(ModelBuilder):
    pass
