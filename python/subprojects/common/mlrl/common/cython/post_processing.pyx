"""
@author Michael Rapp (michael.rapp.ml@gmail.com)
"""
from libcpp.memory cimport make_unique


cdef class PostProcessorFactory:
    """
    A wrapper for the pure virtual C++ class `IPostProcessorFactory`.
    """
    pass


cdef class NoPostProcessorFactory(PostProcessorFactory):
    """
    A wrapper for the C++ class `NoPostProcessorFactory`.
    """

    def __cinit__(self):
        self.post_processor_factory_ptr = <unique_ptr[IPostProcessorFactory]>make_unique[NoPostProcessorFactoryImpl]()
