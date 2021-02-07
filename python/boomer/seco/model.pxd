from boomer.common.model cimport IModelBuilder, ModelBuilder


cdef extern from "cpp/model/decision_list.hpp" namespace "seco" nogil:

    cdef cppclass DecisionListBuilderImpl"seco::DecisionListBuilder"(IModelBuilder):
        pass


cdef class DecisionListBuilder(ModelBuilder):
    pass
