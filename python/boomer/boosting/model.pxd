from boomer.common.model cimport IModelBuilder, ModelBuilder


cdef extern from "cpp/model/rule_list.h" nogil:

    cdef cppclass RuleListBuilderImpl"RuleListBuilder"(IModelBuilder):
        pass


cdef class RuleListBuilder(ModelBuilder):
    pass
