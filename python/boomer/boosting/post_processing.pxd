from boomer.common._types cimport float64
from boomer.common.post_processing cimport PostProcessor, IPostProcessor

from libcpp.memory cimport shared_ptr


cdef extern from "cpp/post_processing.h" namespace "boosting" nogil:

    cdef cppclass ConstantShrinkageImpl(IPostProcessor):

        # Constructors:

        ConstantShrinkageImpl(float64 shrinkage) except +


cdef class ConstantShrinkage(PostProcessor):
    pass
