"""
@author Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides Cython wrappers for C++ classes that allow to calculate the predictions of rules, as well as corresponding
quality scores.
"""
from libcpp.memory cimport make_shared


cdef class LabelWiseRuleEvaluationFactory:
    """
    A wrapper for the pure virtual C++ class `ILabelWiseRuleEvaluationFactory`.
    """
    pass


cdef class RegularizedLabelWiseRuleEvaluationFactory(LabelWiseRuleEvaluationFactory):
    """
    A wrapper for the C++ class `RegularizedLabelWiseRuleEvaluationFactory`.
    """

    def __cinit__(self, float64 l2_regularization_weight):
        """
        :param l2_regularization_weight: The weight of the L2 regularization that is applied for calculating the scores
                                         to be predicted by rules
        """
        self.rule_evaluation_factory_ptr = <shared_ptr[ILabelWiseRuleEvaluationFactory]>make_shared[RegularizedLabelWiseRuleEvaluationFactoryImpl](
            l2_regularization_weight)


cdef class EqualWidthBinningLabelWiseRuleEvaluationFactory(LabelWiseRuleEvaluationFactory):
    """
    A wrapper for the C++ class `EqualWidthBinningLabelWiseRuleEvaluationFactory`.
    """

    def __cinit__(self, float64 l2_regularization_weight, float32 bin_ratio):
        """
        :param l2_regularization_weight:    The weight of the L2 regularization that is applied for calculating the
                                            scores to be predicted by rules
        :param bin_ratio:                   A percentage that specifies how many bins should be used to assign labels to
        """
        self.rule_evaluation_factory_ptr = <shared_ptr[ILabelWiseRuleEvaluationFactory]>make_shared[EqualWidthBinningLabelWiseRuleEvaluationFactoryImpl](
            l2_regularization_weight, bin_ratio)
