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
    A wrapper for the C++ class `RegularizedLabelWiseRuleEvaluationFactoryImpl`.
    """

    def __cinit__(self, float64 l2_regularization_weight):
        """
        :param l2_regularization_weight: The weight of the L2 regularization that is applied for calculating the scores
                                         to be predicted by rules
        """
        self.rule_evaluation_factory_ptr = <shared_ptr[ILabelWiseRuleEvaluationFactory]>make_shared[RegularizedLabelWiseRuleEvaluationFactoryImpl](
            l2_regularization_weight)


cdef class BinningLabelWiseRuleEvaluationFactory(LabelWiseRuleEvaluationFactory):
    """
    A wrapper for the C++ class `BinningLabelWiseRuleEvaluationFactoryImpl`.
    """

    def __cinit__(self, float64 l2_regularization_weight, uint32 num_positive_bins, uint32 num_negative_bins):
        """
        :param l2_regularization_weight:    The weight of the L2 regularization that is applied for calculating the
                                            scores to be predicted by rules
        :param num_positive_bins:           The number of bins to be used for labels that should be predicted as
                                            positive
        :param num_negative_bins:           The number of bins to be used for labels that should be predicted as
                                            negative
        """
        self.rule_evaluation_factory_ptr = <shared_ptr[ILabelWiseRuleEvaluationFactory]>make_shared[BinningLabelWiseRuleEvaluationFactoryImpl](
            l2_regularization_weight, num_positive_bins, num_negative_bins)
