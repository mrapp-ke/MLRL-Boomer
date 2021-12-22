"""
@author: Michael Rapp (michael.rapp.ml@gmail.com)
"""


cdef class SimilarityMeasureFactory:
    """
    A wrapper for the pure virtual C++ class `ISimilarityMeasureFactory`.
    """

    cdef unique_ptr[ISimilarityMeasureFactory] get_similarity_measure_factory_ptr(self):
        pass


cdef class EvaluationMeasureFactory(SimilarityMeasureFactory):
    """
    A wrapper for the pure virtual C++ class `IEvaluationMeasure`.
    """

    cdef unique_ptr[IEvaluationMeasureFactory] get_evaluation_measure_factory_ptr(self):
        pass
