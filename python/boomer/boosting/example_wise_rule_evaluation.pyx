"""
@author Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides classes that allow to calculate the predictions of rules, as well as corresponding quality scores, such that
they minimize a loss function that is applied example-wise.
"""
from boomer.boosting._blas cimport init_blas
from boomer.boosting._lapack cimport init_lapack

from libcpp.memory cimport unique_ptr, make_shared
from libcpp.utility cimport move


cdef class ExampleWiseRuleEvaluation:
    """
    A wrapper for the pure virtual C++ class `IExampleWiseRuleEvaluation`.
    """
    pass


cdef class RegularizedExampleWiseRuleEvaluation(ExampleWiseRuleEvaluation):
    """
    A wrapper for the C++ class `RegularizedExampleWiseRuleEvaluationImpl`.
    """

    def __cinit__(self, float64 l2_regularization_weight):
        """
        :param l2_regularization_weight: The weight of the L2 regularization that is applied for calculating the scores
                                         to be predicted by rules
        """
        cdef unique_ptr[Blas] blas_ptr = init_blas()
        cdef unique_ptr[Lapack] lapack_ptr = init_lapack()
        self.rule_evaluation_ptr = <shared_ptr[IExampleWiseRuleEvaluation]>make_shared[RegularizedExampleWiseRuleEvaluationImpl](
            l2_regularization_weight, move(blas_ptr), move(lapack_ptr))
