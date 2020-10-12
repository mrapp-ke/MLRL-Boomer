"""
@author Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides classes that allow to post-process the predictions of rules once they have been learned.
"""
from libcpp.memory cimport make_shared


cdef class PostProcessor:
    """
    A wrapper for the pure virtual C++ class `IPostProcessor`.
    """
    pass


cdef class NoPostProcessor(PostProcessor):
    """
    A wrapper for the C++ class `NoPostProcessorImpl`.
    """

    def __cinit__(self):
        self.post_processor_ptr = <shared_ptr[IPostProcessor]>make_shared[NoPostProcessorImpl]()
