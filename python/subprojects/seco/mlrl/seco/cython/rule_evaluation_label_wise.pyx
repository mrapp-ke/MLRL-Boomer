"""
@author Jakob Steeg (jakob.steeg@gmail.com)
@author Michael Rapp (michael.rapp.ml@gmail.com)
"""
from mlrl.seco.cython.heuristics cimport HeuristicFactory
from mlrl.seco.cython.lift_functions cimport LiftFunctionFactory

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


cdef class LabelWisePartialRuleEvaluationFactory(LabelWiseRuleEvaluationFactory):
    """
    A wrapper for the C++ class `LabelWisePartialRuleEvaluationFactory`.
    """

    def __cinit__(self, HeuristicFactory heuristic_factory not None,
                  LiftFunctionFactory lift_function_factory not None):
        """
        :param heuristic_factory:       The `HeuristicFactory` that should be used
        :param liftFunction_factory:    The `LiftFunctionFactory` that should be used
        """
        self.rule_evaluation_factory_ptr = <unique_ptr[ILabelWiseRuleEvaluationFactory]>make_unique[LabelWisePartialRuleEvaluationFactoryImpl](
            move(heuristic_factory.heuristic_factory_ptr), move(lift_function_factory.lift_function_factory_ptr))


cdef class LabelWiseSingleLabelRuleEvaluationFactory(LabelWiseRuleEvaluationFactory):
    """
    A wrapper for the C++ class `LabelWiseSingleLabelRuleEvaluationFactory`.
    """

    def __cinit__(self, HeuristicFactory heuristic_factory not None):
        """
        :param heuristic_factory: The `HeuristicFactory` that should be used
        """
        self.rule_evaluation_factory_ptr = <unique_ptr[ILabelWiseRuleEvaluationFactory]>make_unique[LabelWiseSingleLabelRuleEvaluationFactoryImpl](
            move(heuristic_factory.heuristic_factory_ptr))
