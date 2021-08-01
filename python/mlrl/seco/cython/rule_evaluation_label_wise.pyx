"""
@author Jakob Steeg (jakob.steeg@gmail.com)
@author Michael Rapp (mrapp@ke.tu-darmstadt.de)
"""
from mlrl.seco.cython.heuristics cimport Heuristic

from libcpp.utility cimport move
from libcpp.memory cimport make_unique


cdef class LabelWiseRuleEvaluationFactory:
    """
    A wrapper for the pure virtual C++ class `ILabelWiseRuleEvaluationFactory`.
    """
    pass


cdef class HeuristicLabelWiseRuleEvaluationFactory(LabelWiseRuleEvaluationFactory):
    """
    A wrapper for the C++ class `HeuristicLabelWiseRuleEvaluationFactory`.
    """

    def __cinit__(self, Heuristic heuristic not None, bint predictMajority = False):
        """
        :param heuristic:       The heuristic that should be used
        :param predictMajority: True, if for each label the majority label should be predicted, False, if the minority
                                label should be predicted
        """
        self.rule_evaluation_factory_ptr = <unique_ptr[ILabelWiseRuleEvaluationFactory]>make_unique[HeuristicLabelWiseRuleEvaluationFactoryImpl](
            move(heuristic.heuristic_ptr), predictMajority)
