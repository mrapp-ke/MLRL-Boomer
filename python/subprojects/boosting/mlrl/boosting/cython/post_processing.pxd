from mlrl.common.cython._types cimport float64
from mlrl.common.cython.post_processing cimport PostProcessorFactory, IPostProcessorFactory


cdef extern from "boosting/post_processing/shrinkage_constant.hpp" namespace "boosting" nogil:

    cdef cppclass ConstantShrinkageFactoryImpl"boosting::ConstantShrinkageFactory"(IPostProcessorFactory):

        # Constructors:

        ConstantShrinkageImpl(float64 shrinkage) except +


cdef class ConstantShrinkageFactory(PostProcessorFactory):
    pass
