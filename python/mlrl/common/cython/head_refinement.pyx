"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)
"""
from libcpp.memory cimport make_unique


cdef class HeadRefinementFactory:
    """
    A wrapper for the pure virtual C++ class `IHeadRefinementFactory`.
    """
    pass


cdef class SingleLabelHeadRefinementFactory(HeadRefinementFactory):
    """
    A wrapper for the C++ class `SingleLabelHeadRefinementFactory`.
    """

    def __cinit__(self):
        self.head_refinement_factory_ptr = <unique_ptr[IHeadRefinementFactory]>make_unique[SingleLabelHeadRefinementFactoryImpl]()


cdef class CompleteHeadRefinementFactory(HeadRefinementFactory):
    """
    A wrapper for the C++ class `CompleteHeadRefinementFactory`.
    """

    def __cinit__(self):
        self.head_refinement_factory_ptr = <unique_ptr[IHeadRefinementFactory]>make_unique[CompleteHeadRefinementFactoryImpl]()
