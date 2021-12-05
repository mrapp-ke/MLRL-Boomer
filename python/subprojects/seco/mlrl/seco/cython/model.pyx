"""
@author Michael Rapp (michael.rapp.ml@gmail.com)
"""
from libcpp.memory cimport unique_ptr, make_unique


cdef class DecisionListBuilder(ModelBuilder):
    """
    A wrapper for the C++ class `DecisionListBuilder`.
    """

    def __cinit__(self):
        self.model_builder_ptr = <unique_ptr[IModelBuilder]>make_unique[DecisionListBuilderImpl]()
