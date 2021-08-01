"""
@author Michael Rapp (mrapp@ke.tu-darmstadt.de)
"""
from libcpp.memory cimport make_unique
from mlrl.boosting.cython.binning cimport LabelBinningFactory

from libcpp.utility cimport move


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
        self.rule_evaluation_factory_ptr = <unique_ptr[ILabelWiseRuleEvaluationFactory]>make_unique[RegularizedLabelWiseRuleEvaluationFactoryImpl](
            l2_regularization_weight)


cdef class BinnedLabelWiseRuleEvaluationFactory(LabelWiseRuleEvaluationFactory):
    """
    A wrapper for the C++ class `BinnedLabelWiseRuleEvaluationFactory`.
    """

    def __cinit__(self, float64 l2_regularization_weight, LabelBinningFactory label_binning_factory not None):
        """
        :param l2_regularization_weight:    The weight of the L2 regularization that is applied for calculating the
                                            scores to be predicted by rules
        :param label_binning_factory:       A `LabelBinningFactory` that allows to create the implementation that should
                                            be used to assign labebls to bins
        """
        self.rule_evaluation_factory_ptr = <unique_ptr[ILabelWiseRuleEvaluationFactory]>make_unique[BinnedLabelWiseRuleEvaluationFactoryImpl](
            l2_regularization_weight, move(label_binning_factory.label_binning_factory_ptr))
