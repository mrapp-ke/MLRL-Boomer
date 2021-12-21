from libcpp.memory cimport unique_ptr


cdef extern from "common/post_processing/post_processor.hpp" nogil:

    cdef cppclass IPostProcessorFactory:
        pass


cdef extern from "common/post_processing/post_processor_no.hpp" nogil:

    cdef cppclass NoPostProcessorFactoryImpl"NoPostProcessorFactory"(IPostProcessorFactory):
        pass


cdef class PostProcessorFactory:

    # Attributes:

    cdef unique_ptr[IPostProcessorFactory] post_processor_factory_ptr


cdef class NoPostProcessorFactory(PostProcessorFactory):
    pass
