from libcpp.memory cimport shared_ptr


cdef extern from "cpp/post_processing/post_processor.h" nogil:

    cdef cppclass IPostProcessor:
        pass


cdef extern from "cpp/post_processing/post_processor_no.h" nogil:

    cdef cppclass NoPostProcessorImpl"NoPostProcessor"(IPostProcessor):
        pass


cdef class PostProcessor:

    # Attributes:

    cdef shared_ptr[IPostProcessor] post_processor_ptr


cdef class NoPostProcessor(PostProcessor):
    pass
