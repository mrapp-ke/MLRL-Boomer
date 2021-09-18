"""
@author Michael Rapp (mrapp@ke.tu-darmstadt.de)
"""
from mlrl.boosting.cython.label_binning cimport LabelBinningFactory

from libcpp.cast cimport dynamic_cast
from libcpp.memory cimport make_unique
from libcpp.utility cimport move


cdef class LabelWiseRuleEvaluationFactory:
    """
    A wrapper for the pure virtual C++ class `ILabelWiseRuleEvaluationFactory`.
    """
    pass


cdef class SparseLabelWiseRuleEvaluationFactory(LabelWiseRuleEvaluationFactory):
    """
    A wrapper for the pure virtual C++ class `ISparseLabelWiseRuleEvaluationFactory`.
    """

    cdef unique_ptr[ISparseLabelWiseRuleEvaluationFactory] get_sparse_label_wise_rule_rule_evaluation_factory_ptr(self):
        return unique_ptr[ISparseLabelWiseRuleEvaluationFactory](dynamic_cast[ISparseLabelWiseRuleEvaluationFactoryPtr](self.rule_evaluation_factory_ptr.release()))


cdef class LabelWiseSingleLabelRuleEvaluationFactory(SparseLabelWiseRuleEvaluationFactory):
    """
    A wrapper for the C++ class `LabelWiseSingleLabelRuleEvaluationFactory`.
    """

    def __cinit__(self, float64 l2_regularization_weight):
        """
        :param l2_regularization_weight: The weight of the L2 regularization that is applied for calculating the scores
                                         to be predicted by rules
        """
        self.rule_evaluation_factory_ptr = <unique_ptr[ILabelWiseRuleEvaluationFactory]>make_unique[LabelWiseSingleLabelRuleEvaluationFactoryImpl](
            l2_regularization_weight)


cdef class LabelWiseCompleteRuleEvaluationFactory(LabelWiseRuleEvaluationFactory):
    """
    A wrapper for the C++ class `LabelWiseCompleteRuleEvaluationFactory`.
    """

    def __cinit__(self, float64 l2_regularization_weight):
        """
        :param l2_regularization_weight: The weight of the L2 regularization that is applied for calculating the scores
                                         to be predicted by rules
        """
        self.rule_evaluation_factory_ptr = <unique_ptr[ILabelWiseRuleEvaluationFactory]>make_unique[LabelWiseCompleteRuleEvaluationFactoryImpl](
            l2_regularization_weight)


cdef class LabelWiseCompleteBinnedRuleEvaluationFactory(LabelWiseRuleEvaluationFactory):
    """
    A wrapper for the C++ class `LabelWiseCompleteBinnedRuleEvaluationFactory`.
    """

    def __cinit__(self, float64 l2_regularization_weight, LabelBinningFactory label_binning_factory not None):
        """
        :param l2_regularization_weight:    The weight of the L2 regularization that is applied for calculating the
                                            scores to be predicted by rules
        :param label_binning_factory:       A `LabelBinningFactory` that allows to create the implementation that should
                                            be used to assign labels to bins
        """
        self.rule_evaluation_factory_ptr = <unique_ptr[ILabelWiseRuleEvaluationFactory]>make_unique[LabelWiseCompleteBinnedRuleEvaluationFactoryImpl](
            l2_regularization_weight, move(label_binning_factory.label_binning_factory_ptr))
