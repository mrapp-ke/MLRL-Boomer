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


cdef class LabelWiseMajorityRuleEvaluationFactory(LabelWiseRuleEvaluationFactory):
    """
    A wrapper for the C++ class `LabelWiseMajorityRuleEvaluationFactory`.
    """

    def __cinit__(self):
        self.rule_evaluation_factory_ptr = <unique_ptr[ILabelWiseRuleEvaluationFactory]>make_unique[LabelWiseMajorityRuleEvaluationFactoryImpl]()


cdef class LabelWiseSingleLabelRuleEvaluationFactory(LabelWiseRuleEvaluationFactory):
    """
    A wrapper for the C++ class `LabelWiseSingleLabelRuleEvaluationFactory`.
    """

    def __cinit__(self, Heuristic heuristic not None):
        """
        :param heuristic:
        """
        self.rule_evaluation_factory_ptr = <unique_ptr[ILabelWiseRuleEvaluationFactory]>make_unique[LabelWiseSingleLabelRuleEvaluationFactoryImpl](
            move(heuristic.heuristic_ptr))
