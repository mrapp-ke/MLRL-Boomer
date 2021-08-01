from libcpp.memory cimport unique_ptr


cdef extern from "common/head_refinement/head_refinement_factory.hpp" nogil:

    cdef cppclass IHeadRefinementFactory:
        pass


cdef extern from "common/head_refinement/head_refinement_single.hpp" nogil:

    cdef cppclass SingleLabelHeadRefinementFactoryImpl"SingleLabelHeadRefinementFactory"(IHeadRefinementFactory):
        pass


cdef extern from "common/head_refinement/head_refinement_complete.hpp" nogil:

    cdef cppclass CompleteHeadRefinementFactoryImpl"CompleteHeadRefinementFactory"(IHeadRefinementFactory):
        pass


cdef class HeadRefinementFactory:

    # Attributes:

    cdef unique_ptr[IHeadRefinementFactory] head_refinement_factory_ptr


cdef class SingleLabelHeadRefinementFactory(HeadRefinementFactory):
    pass


cdef class CompleteHeadRefinementFactory(HeadRefinementFactory):
    pass
