"""
@author Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides classes that allow to calculate the predictions of rules, as well as corresponding quality scores, such that
they minimize a loss function that is applied example-wise.
"""
from boomer.boosting._blas cimport init_blas
from boomer.boosting._lapack cimport init_lapack

from libcpp.memory cimport make_shared
from libcpp.utility cimport move


cdef class ExampleWiseRuleEvaluation:
    """
    A wrapper for the pure virtual C++ class `IExampleWiseRuleEvaluation`.
    """
    pass


cdef class RegularizedExampleWiseRuleEvaluationFactory(ExampleWiseRuleEvaluationFactory):
    """
    A wrapper for the C++ class `RegularizedExampleWiseRuleEvaluationFactory`.
    """

    def __cinit__(self, float64 l2_regularization_weight):
        """
        :param l2_regularization_weight: The weight of the L2 regularization that is applied for calculating the scores
                                         to be predicted by rules
        """
        cdef shared_ptr[Blas] blas_ptr = <shared_ptr[Blas]>move(init_blas())
        cdef shared_ptr[Lapack] lapack_ptr = <shared_ptr[Lapack]>move(init_lapack())
        self.rule_evaluation_factory_ptr = <shared_ptr[IExampleWiseRuleEvaluationFactory]>make_shared[RegularizedExampleWiseRuleEvaluationFactoryImpl](
            l2_regularization_weight, blas_ptr, lapack_ptr)


cdef class BinningExampleWiseRuleEvaluationFactory(ExampleWiseRuleEvaluationFactory):
    """
    A wrapper for the C++ class `BinningExampleWiseRuleEvaluationFactoryImpl`.
    """

    def __cinit__(self, float64 l2_regularization_weight, uint32 num_positive_bins, uint32 num_negative_bins):
        """
        :param l2_regularization_weight:    The weight of the L2 regularization that is applied for calculating the
                                            scores to be predicted by rules
        :param num_positive_bins:           The number of bins to be used for labels that should be predicted positively
        :param num_negative_bins:           The number of bins to be used for labels that should be predicted negatively
        """
        cdef shared_ptr[Blas] blas_ptr = <shared_ptr[Blas]>move(init_blas())
        cdef shared_ptr[Lapack] lapack_ptr = <shared_ptr[Lapack]>move(init_lapack())
        self.rule_evaluation_factory_ptr = <shared_ptr[IExampleWiseRuleEvaluationFactory]>make_shared[BinningExampleWiseRuleEvaluationFactoryImpl](
            l2_regularization_weight, num_positive_bins, num_negative_bins, blas_ptr, lapack_ptr)
