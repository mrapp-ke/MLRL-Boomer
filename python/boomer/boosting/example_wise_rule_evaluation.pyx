"""
@author Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides classes that allow to calculate the predictions of rules, as well as corresponding quality scores, such that
they minimize a loss function that is applied example-wise.
"""
from boomer.boosting._blas cimport init_blas
from boomer.boosting._lapack cimport init_lapack
from boomer.boosting.example_wise_losses cimport ExampleWiseLoss

from libcpp.memory cimport make_shared


cdef class ExampleWiseDefaultRuleEvaluation(DefaultRuleEvaluation):
    """
    A wrapper for the C++ class `ExampleWiseDefaultRuleEvaluationImpl`.
    """

    def __cinit__(self, ExampleWiseLoss loss_function, float64 l2_regularization_weight):
        """
        :param loss_function:               The loss function to be minimized
        :param l2_regularization_weight:    The weight of the L2 regularization that is applied for calculating the
                                            scores to be predicted by the default rule
        """
        cdef shared_ptr[AbstractExampleWiseLoss] loss_function_ptr = loss_function.loss_function_ptr
        cdef shared_ptr[Lapack] lapack_ptr = init_lapack()
        self.default_rule_evaluation_ptr = <shared_ptr[AbstractDefaultRuleEvaluation]>make_shared[ExampleWiseDefaultRuleEvaluationImpl](
            loss_function_ptr, l2_regularization_weight, lapack_ptr)


cdef class ExampleWiseRuleEvaluation:
    """
    A wrapper for the C++ class `ExampleWiseRuleEvaluationImpl`.
    """

    def __cinit__(self, float64 l2_regularization_weight):
        """
        :param l2_regularization_weight: The weight of the L2 regularization that is applied for calculating the scores
                                         to be predicted by rules
        """
        cdef shared_ptr[Blas] blas_ptr = init_blas()
        cdef shared_ptr[Lapack] lapack_ptr = init_lapack()
        self.rule_evaluation_ptr = make_shared[ExampleWiseRuleEvaluationImpl](l2_regularization_weight, blas_ptr,
                                                                              lapack_ptr)
