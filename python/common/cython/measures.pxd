from common.cython._measures cimport ISimilarityMeasure

from libcpp.memory cimport shared_ptr


cdef class SimilarityMeasure:

    # Functions:

    cdef shared_ptr[ISimilarityMeasure] get_similarity_measure_ptr(self)
