from boomer.common.model cimport IModelBuilder, ModelBuilder


cdef extern from "cpp/model/decision_list.h" nogil:

    cdef cppclass DecisionListBuilderImpl"DecisionListBuilder"(IModelBuilder):
        pass


cdef class DecisionListBuilder(ModelBuilder):
    pass
