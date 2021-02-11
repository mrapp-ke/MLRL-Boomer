"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)
"""


cdef class SimilarityMeasure:
    """
    A wrapper for the pure virtual C++ class `ISimilarityMeasure`.
    """

    cdef shared_ptr[ISimilarityMeasure] get_similarity_measure_ptr(self):
        pass
