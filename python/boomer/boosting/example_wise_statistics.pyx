"""
@author Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides wrappers for classes that allow to store gradients and Hessians that are calculated according to a
(non-decomposable) loss function that is applied example-wise.
"""
from boomer.boosting._lapack cimport init_lapack
from boomer.boosting.example_wise_losses cimport ExampleWiseLoss
from boomer.boosting.example_wise_rule_evaluation cimport ExampleWiseRuleEvaluation

from libcpp.memory cimport make_shared


cdef class ExampleWiseStatistics(GradientStatistics):
    """
    A wrapper for the class `ExampleWiseStatisticsImpl`.
    """

    def __cinit__(self, ExampleWiseLoss loss_function, ExampleWiseRuleEvaluation rule_evaluation):
        """
        :param loss_function:   The loss function to be used for calculating gradients and Hessians
        :param rule_evaluation: The `ExampleWiseRuleEvaluation` to be used for calculating the predictions, as well as
                                corresponding quality scores, of rules
        """
        cdef shared_ptr[AbstractExampleWiseLoss] loss_function_ptr = loss_function.loss_function_ptr
        cdef shared_ptr[ExampleWiseRuleEvaluationImpl] rule_evaluation_ptr = rule_evaluation.rule_evaluation_ptr
        cdef shared_ptr[Lapack] lapack_ptr
        lapack_ptr.reset(init_lapack())
        self.statistics_ptr = <shared_ptr[AbstractStatistics]>make_shared[ExampleWiseStatisticsImpl](
            loss_function_ptr, rule_evaluation_ptr, lapack_ptr)
