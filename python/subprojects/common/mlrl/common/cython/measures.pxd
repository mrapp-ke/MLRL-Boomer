from mlrl.common.cython._measures cimport IEvaluationMeasureFactory, ISimilarityMeasureFactory

from libcpp.memory cimport unique_ptr


cdef class SimilarityMeasureFactory:

    # Functions:

    cdef unique_ptr[ISimilarityMeasureFactory] get_similarity_measure_factory_ptr(self)


cdef class EvaluationMeasureFactory(SimilarityMeasureFactory):

    # Functions:

    cdef unique_ptr[IEvaluationMeasureFactory] get_evaluation_measure_factory_ptr(self)
