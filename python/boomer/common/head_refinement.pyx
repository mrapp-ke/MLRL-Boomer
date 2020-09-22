"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides classes that implement strategies for finding the heads of rules.
"""
from libcpp.memory cimport make_shared


cdef class HeadRefinement:
    """
    A wrapper for the pure virtual C++ class `IHeadRefinement`.
    """
    pass


cdef class SingleLabelHeadRefinement(HeadRefinement):
    """
    A wrapper for the C++ class `SingleLabelHeadRefinementImpl`.
    """

    def __cinit__(self):
        self.head_refinement_ptr = <shared_ptr[IHeadRefinement]>make_shared[SingleLabelHeadRefinementImpl]()


cdef class FullHeadRefinement(HeadRefinement):
    """
    A wrapper for the C++ class `FullHeadRefinementImpl`.
    """

    def __cinit__(self):
        self.head_refinement_ptr = <shared_ptr[IHeadRefinement]>make_shared[FullHeadRefinementImpl]()
