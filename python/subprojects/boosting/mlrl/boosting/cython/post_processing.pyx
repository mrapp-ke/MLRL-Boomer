"""
@author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from libcpp.memory cimport unique_ptr, make_unique


cdef class ConstantShrinkageFactory(PostProcessorFactory):
    """
    A wrapper for the C++ class `ConstantShrinkageFactory`.
    """

    def __cinit__(self, float64 shrinkage):
        """
        :param shrinkage: The shrinkage parameter. Must be in (0, 1)
        """
        self.post_processor_factory_ptr = <unique_ptr[IPostProcessorFactory]>make_unique[ConstantShrinkageFactoryImpl](
            shrinkage)
