from boomer.common._predictions cimport Prediction

from libcpp.memory cimport shared_ptr


cdef extern from "cpp/post_processing.h" nogil:

    cdef cppclass IPostProcessor:

        # Functions:

        void postProcess(Prediction& prediction)


    cdef cppclass NoPostProcessorImpl(IPostProcessor):
        pass


cdef class PostProcessor:

    # Attributes:

    cdef shared_ptr[IPostProcessor] post_processor_ptr


cdef class NoPostProcessor(PostProcessor):
    pass
