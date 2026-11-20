"""
@author Michael Rapp (michael.rapp.ml@gmail.com)
"""
from libcpp.cast cimport dynamic_cast

from cython.operator cimport dereference

from mlrl.common.cython._serialization cimport check_pickled_state, image_to_bytes

from abc import abstractmethod
from dataclasses import dataclass
from enum import StrEnum

import numpy as np


class Comparator(StrEnum):
    """
    The comparison operators that can be used by a condition of a rule.
    """
    LEQ = '<='
    GR = '>'
    EQ = '=='
    NEQ = '!='


@dataclass
class Condition:
    """
    A single condition contained in the body of a rule.

    Attributes:
        feature_index:  The index of the feature, the condition corresponds to
        comparator:     The comparison operator that is used by the condition
        threshold:      The threshold that is used by the condition
    """
    feature_index: int
    comparator: Comparator
    threshold: int | float


cdef class Body:
    """
    An abstract base class for all bodies of a rule.
    """


cdef class EmptyBody(Body):
    """
    A body of a rule that does not contain any conditions.
    """

    def __iter__(self) -> Iterator[Condition]:
        yield from []


cdef class ConjunctiveBody(Body):
    """
    A body of a rule that is given as a conjunction of several conditions.
    """

    def __cinit__(self, const uint32[::1] numerical_leq_indices, const float32[::1] numerical_leq_thresholds,
                  const uint32[::1] numerical_gr_indices, const float32[::1] numerical_gr_thresholds,
                  const uint32[::1] ordinal_leq_indices, const int32[::1] ordinal_leq_thresholds,
                  const uint32[::1] ordinal_gr_indices, const int32[::1] ordinal_gr_thresholds,
                  const uint32[::1] nominal_eq_indices, const int32[::1] nominal_eq_thresholds,
                  const uint32[::1] nominal_neq_indices, const int32[::1] nominal_neq_thresholds):
        """
        :param numerical_leq_indices:       A contiguous array of type `uint32`, shape `(num_numerical_leq_conditions)`,
                                            that stores the feature indices of the numerical conditions that use the <=
                                            operator or None, if no such conditions are available
        :param numerical_leq_thresholds:    A contiguous array of type `float32`, shape `(num_numerical_leq_conditions)`
                                            that stores the thresholds of the numerical conditions that use the <=
                                            operator or None, if no such conditions are available
        :param numerical_gr_indices:        A contiguous array of type `uint32`, shape `(num_numerical_gr_conditions)`,
                                            that stores the feature indices of the numerical conditions that use the >
                                            operator or None, if no such conditions are available
        :param numerical_gr_thresholds:     A contiguous array of type `float32`, shape `(num_numerical_gr_conditions)`
                                            that stores the thresholds of the numerical conditions that use the >
                                            operator or None, if no such conditions are available
        :param ordinal_leq_indices:         A contiguous array of type `uint32`, shape `(num_ordinal_leq_conditions)`,
                                            that stores the feature indices of the ordinal conditions that use the <=
                                            operator or None, if no such conditions are available
        :param ordinal_leq_thresholds:      A contiguous array of type `int32`, shape `(num_ordinal_leq_conditions)`
                                            that stores the thresholds of the ordinal conditions that use the <=
                                            operator or None, if no such conditions are available
        :param ordinal_gr_indices:          A contiguous array of type `uint32`, shape `(num_ordinal_gr_conditions)`,
                                            that stores the feature indices of the ordinal conditions that use the >
                                            operator or None, if no such conditions are available
        :param ordinal_gr_thresholds:       A contiguous array of type `int32`, shape `(num_ordinal_gr_conditions)` that
                                            stores the thresholds of the ordinal conditions that use the > operator or
                                            None, if no such conditions are available
        :param nominal_eq_indices:          A contiguous array of type `uint32`, shape `(num_nominal_eq_conditions)`,
                                            that stores the feature indices of the nominal conditions that use the ==
                                            operator or None, if no such conditions are available
        :param nominal_eq_thresholds:       A contiguous array of type `int32`, shape `(num_nominal_eq_conditions)` that
                                            stores the thresholds of the nominal conditions that use the == operator or
                                            None, if no such conditions are available
        :param nominal_neq_indices:         A contiguous array of type `uint32`, shape `(num_nominal_neq_conditions)`,
                                            that stores the feature indices of the nominal conditions that use the !=
                                            operator or None, if no such conditions are available
        :param nominal_neq_thresholds:      A contiguous array of type `int32`, shape `(num_nominal_neq_conditions)`
                                            that stores the thresholds of the nominal conditions that use the !=
                                            operator or None, if no such conditions are available
        """
        self.numerical_leq_indices = np.asarray(numerical_leq_indices) if numerical_leq_indices is not None else None
        self.numerical_leq_thresholds = \
            np.asarray(numerical_leq_thresholds) if numerical_leq_thresholds is not None else None
        self.numerical_gr_indices = np.asarray(numerical_gr_indices) if numerical_gr_indices is not None else None
        self.numerical_gr_thresholds = \
            np.asarray(numerical_gr_thresholds) if numerical_gr_thresholds is not None else None
        self.ordinal_leq_indices = np.asarray(ordinal_leq_indices) if ordinal_leq_indices is not None else None
        self.ordinal_leq_thresholds = np.asarray(ordinal_leq_thresholds) if ordinal_leq_thresholds is not None else None
        self.ordinal_gr_indices = np.asarray(ordinal_gr_indices) if ordinal_gr_indices is not None else None
        self.ordinal_gr_thresholds = np.asarray(ordinal_gr_thresholds) if ordinal_gr_thresholds is not None else None
        self.nominal_eq_indices = np.asarray(nominal_eq_indices) if nominal_eq_indices is not None else None
        self.nominal_eq_thresholds = np.asarray(nominal_eq_thresholds) if nominal_eq_thresholds is not None else None
        self.nominal_neq_indices = np.asarray(nominal_neq_indices) if nominal_neq_indices is not None else None
        self.nominal_neq_thresholds = np.asarray(nominal_neq_thresholds) if nominal_neq_thresholds is not None else None

    def __iter__(self) -> Iterator[Condition]:
        cdef npc.ndarray indices = self.numerical_leq_indices
        cdef npc.ndarray thresholds = self.numerical_leq_thresholds
        cdef uint32 num_conditions = 0 if indices is None else indices.shape[0]
        cdef uint32 i

        for i in range(num_conditions):
            yield Condition(feature_index=indices[i], comparator=Comparator.LEQ, threshold=thresholds[i])

        indices = self.numerical_gr_indices
        thresholds = self.numerical_gr_thresholds
        num_conditions = 0 if indices is None else indices.shape[0]

        for i in range(num_conditions):
            yield Condition(feature_index=indices[i], comparator=Comparator.GR, threshold=thresholds[i])

        indices = self.ordinal_leq_indices
        thresholds = self.ordinal_leq_thresholds
        num_conditions = 0 if indices is None else indices.shape[0]

        for i in range(num_conditions):
            yield Condition(feature_index=indices[i], comparator=Comparator.LEQ, threshold=thresholds[i])

        indices = self.ordinal_gr_indices
        thresholds = self.ordinal_gr_thresholds
        num_conditions = 0 if indices is None else indices.shape[0]

        for i in range(num_conditions):
            yield Condition(feature_index=indices[i], comparator=Comparator.GR, threshold=thresholds[i])

        indices = self.nominal_eq_indices
        thresholds = self.nominal_eq_thresholds
        num_conditions = 0 if indices is None else indices.shape[0]

        for i in range(num_conditions):
            yield Condition(feature_index=indices[i], comparator=Comparator.EQ, threshold=thresholds[i])

        indices = self.nominal_neq_indices
        thresholds = self.nominal_neq_thresholds
        num_conditions = 0 if indices is None else indices.shape[0]

        for i in range(num_conditions):
            yield Condition(feature_index=indices[i], comparator=Comparator.NEQ, threshold=thresholds[i])


@dataclass
class Prediction:
    """
    A scalar value predicted by a rule for a specific output.

    Attributes:
        output_index:   The index of the output, the prediction corresponds to
        value:          The predicted value
    """
    output_index: int
    value: int | float


cdef class Head:
    """
    An abstract base class for all heads of a rule.
    """


cdef class CompleteHead(Head):
    """
    A head of a rule that predicts numerical scores for all available outputs.
    """

    def __cinit__(self, npc.ndarray scores not None):
        """
        :param scores: A `npc.ndarray`, shape `(num_predictions)` that stores the predicted scores
        """
        self.scores = scores

    def __iter__(self) -> Iterator[Prediction]:
        cdef npc.ndarray scores = self.scores
        cdef uint32 num_predictions = scores.shape[0]
        cdef uint32 output_index

        for output_index in range(num_predictions):
            yield Prediction(output_index=output_index, value=scores[output_index])


cdef class PartialHead(Head):
    """
    A head of a rule that predicts numerical scores for a subset of the available outputs.
    """

    def __cinit__(self, npc.ndarray indices not None, npc.ndarray scores not None):
        """
        :param indices: A `npc.ndarray`, shape `(num_predictions)` that stores the output indices
        :param scores:  A `npc.ndarray`, shape `(num_predictions)` that stores the predicted scores
        """
        self.indices = indices
        self.scores = scores

    def __iter__(self) -> Iterator[Prediction]:
        cdef npc.ndarray indices = self.indices
        cdef npc.ndarray scores = self.scores
        cdef uint32 num_predictions = indices.shape[0]
        cdef uint32 i

        for i in range(num_predictions):
            yield Prediction(output_index=indices[i], value=scores[i])


cdef class Rule:
    """
    A single rule, consisting of a body and a head.
    """

    def __cinit__(self, Body body not None, Head head not None):
        """
        :param body:    The body of the rule
        :param head:    The head of the rule
        """
        self.body = body
        self.head = head


class RuleModelVisitor:
    """
    Defines the methods that must be implemented by a visitor that accesses the bodies and heads of the rules in a
    rule-based model according to the visitor pattern.
    """

    @abstractmethod
    def visit_empty_body(self, body: EmptyBody):
        """
        Must be implemented by subclasses in order to visit bodies of rules that do not contain any conditions.

        :param body: An `EmptyBody` to be visited
        """

    @abstractmethod
    def visit_conjunctive_body(self, body: ConjunctiveBody):
        """
        Must be implemented by subclasses in order to visit the bodies of rule that are given as a conjunction of
        several conditions.

        :param body: A `ConjunctiveBody` to be visited
        """

    @abstractmethod
    def visit_complete_head(self, head: CompleteHead):
        """
        Must be implemented by subclasses in order to visit the heads of rules that predict numerical scores for all
        available outputs.

        :param head: A `CompleteHead` to be visited
        """

    @abstractmethod
    def visit_partial_head(self, head: PartialHead):
        """
        Must be implemented by subclasses in order to visit the heads of rules that predict numerical scores for a
        subset of the available outputs.

        :param head: A `PartialHead` to be visited
        """


cdef class RuleModel:
    """
    A rule-based model.
    """

    cdef IRuleModel* get_rule_model_ptr(self):
        pass

    def get_num_rules(self) -> int:
        """
        Returns the total number of rules in the model, including the default rule, if available.

        :return The total number of rules in the model
        """
        return self.get_rule_model_ptr().getNumRules()

    def get_num_used_rules(self) -> int:
        """
        Returns the number of used rules in the model, including the default rule, if available.

        :return The number of used rules in the model
        """
        return self.get_rule_model_ptr().getNumUsedRules()

    def set_num_used_rules(self, num_used_rules: int):
        """
        Sets the number of used rules in the model, including the default rule, if available.

        :param num_used_rules: The number of used rules to be set
        """
        self.get_rule_model_ptr().setNumUsedRules(num_used_rules)

    def __iter__(self) -> Iterator[Rule]:
        class Visitor(RuleModelVisitor):
            """
            Yields the rules in the model.
            """

            def __init__(self):
                self._current_body = None
                self.rules = []

            def visit_empty_body(self, body: EmptyBody):
                self._current_body = body

            def visit_conjunctive_body(self, body: ConjunctiveBody):
                self._current_body = body

            def visit_complete_head(self, head: CompleteHead):
                self.rules.append(Rule.__new__(Rule, self._current_body, head))

            def visit_partial_head(self, head: PartialHead):
                self.rules.append(Rule.__new__(Rule, self._current_body, head))

        if self.get_num_used_rules() > 0:
            visitor = Visitor()
            self.visit_used(visitor)
            yield from visitor.rules
        else:
            yield from []

    @abstractmethod
    def visit(self, visitor: RuleModelVisitor):
        """
        Visits the bodies and heads of all rules that are contained in this model, including the default rule, if
        available.

        :param visitor: The `RuleModelVisitor` that should be used to access the bodies and heads
        """

    @abstractmethod
    def visit_used(self, visitor: RuleModelVisitor):
        """
        Visits the bodies and heads of all used rules that are contained in this model, including the default rule, if
        available.

        :param visitor: The `RuleModelVisitor` that should be used to access the bodies and heads
        """


cdef class RuleList(RuleModel):
    """
    A rule-based model that stores several rules in an ordered list.
    """

    def __cinit__(self):
        self.visitor = None

    cdef IRuleModel* get_rule_model_ptr(self):
        return self.rule_list_ptr.get()

    cdef __visit_empty_body(self, const EmptyBodyImpl& body):
        self.visitor.visit_empty_body(EmptyBody.__new__(EmptyBody))

    cdef __visit_conjunctive_body(self, const ConjunctiveBodyImpl& body):
        cdef uint32 num_numerical_leq = body.getNumNumericalLeq()
        cdef const uint32[::1] numerical_leq_indices = \
            <uint32[:num_numerical_leq]>body.numerical_leq_indices_cbegin() if num_numerical_leq > 0 else None
        cdef const float32[::1] numerical_leq_thresholds = \
            <float32[:num_numerical_leq]>body.numerical_leq_thresholds_cbegin() if num_numerical_leq > 0 else None

        cdef uint32 num_numerical_gr = body.getNumNumericalGr()
        cdef const uint32[::1] numerical_gr_indices = \
            <uint32[:num_numerical_gr]>body.numerical_gr_indices_cbegin() if num_numerical_gr > 0 else None
        cdef const float32[::1] numerical_gr_thresholds = \
            <float32[:num_numerical_gr]>body.numerical_gr_thresholds_cbegin() if num_numerical_gr > 0 else None

        cdef uint32 num_ordinal_leq = body.getNumOrdinalLeq()
        cdef const uint32[::1] ordinal_leq_indices = \
            <uint32[:num_ordinal_leq]>body.ordinal_leq_indices_cbegin() if num_ordinal_leq > 0 else None
        cdef const int32[::1] ordinal_leq_thresholds = \
            <int32[:num_ordinal_leq]>body.ordinal_leq_thresholds_cbegin() if num_ordinal_leq > 0 else None

        cdef uint32 num_ordinal_gr = body.getNumOrdinalGr()
        cdef const uint32[::1] ordinal_gr_indices = \
            <uint32[:num_ordinal_gr]>body.ordinal_gr_indices_cbegin() if num_ordinal_gr > 0 else None
        cdef const int32[::1] ordinal_gr_thresholds = \
            <int32[:num_ordinal_gr]>body.ordinal_gr_thresholds_cbegin() if num_ordinal_gr > 0 else None

        cdef uint32 num_nominal_eq = body.getNumNominalEq()
        cdef const uint32[::1] nominal_eq_indices = \
            <uint32[:num_nominal_eq]>body.nominal_eq_indices_cbegin() if num_nominal_eq > 0 else None
        cdef const int32[::1] nominal_eq_thresholds = \
            <int32[:num_nominal_eq]>body.nominal_eq_thresholds_cbegin() if num_nominal_eq > 0 else None

        cdef uint32 num_nominal_neq = body.getNumNominalNeq()
        cdef const uint32[::1] nominal_neq_indices = \
            <uint32[:num_nominal_neq]>body.nominal_neq_indices_cbegin() if num_nominal_neq > 0 else None
        cdef const int32[::1] nominal_neq_thresholds = \
            <int32[:num_nominal_neq]>body.nominal_neq_thresholds_cbegin() if num_nominal_neq > 0 else None

        self.visitor.visit_conjunctive_body(
            ConjunctiveBody.__new__(ConjunctiveBody, numerical_leq_indices, numerical_leq_thresholds,
                                    numerical_gr_indices, numerical_gr_thresholds, ordinal_leq_indices,
                                    ordinal_leq_thresholds, ordinal_gr_indices, ordinal_gr_thresholds,
                                    nominal_eq_indices, nominal_eq_thresholds, nominal_neq_indices,
                                    nominal_neq_thresholds))

    cdef __visit_complete_binary_head(self, const CompleteBinaryHeadImpl& head):
        cdef uint32 num_elements = head.getNumElements()
        cdef npc.ndarray values = np.asarray(<uint8[:num_elements]>head.values_cbegin())
        self.visitor.visit_complete_head(CompleteHead.__new__(CompleteHead, values))

    cdef __visit_complete_32bit_head(self, const Complete32BitHeadImpl& head):
        cdef uint32 num_elements = head.getNumElements()
        cdef npc.ndarray values = np.asarray(<float32[:num_elements]>head.values_cbegin())
        self.visitor.visit_complete_head(CompleteHead.__new__(CompleteHead, values))

    cdef __visit_complete_64bit_head(self, const Complete64BitHeadImpl& head):
        cdef uint32 num_elements = head.getNumElements()
        cdef npc.ndarray values = np.asarray(<float64[:num_elements]>head.values_cbegin())
        self.visitor.visit_complete_head(CompleteHead.__new__(CompleteHead, values))

    cdef __visit_partial_binary_head(self, const PartialBinaryHeadImpl& head):
        cdef uint32 num_elements = head.getNumElements()
        cdef npc.ndarray indices = np.asarray(<uint32[:num_elements]>head.indices_cbegin())
        cdef npc.ndarray values = np.asarray(<uint8[:num_elements]>head.values_cbegin())
        self.visitor.visit_partial_head(PartialHead.__new__(PartialHead, indices, values))

    cdef __visit_partial_32bit_head(self, const Partial32BitHeadImpl& head):
        cdef uint32 num_elements = head.getNumElements()
        cdef npc.ndarray indices = np.asarray(<uint32[:num_elements]>head.indices_cbegin())
        cdef npc.ndarray values = np.asarray(<float32[:num_elements]>head.values_cbegin())
        self.visitor.visit_partial_head(PartialHead.__new__(PartialHead, indices, values))

    cdef __visit_partial_64bit_head(self, const Partial64BitHeadImpl& head):
        cdef uint32 num_elements = head.getNumElements()
        cdef npc.ndarray indices = np.asarray(<uint32[:num_elements]>head.indices_cbegin())
        cdef npc.ndarray values = np.asarray(<float64[:num_elements]>head.values_cbegin())
        self.visitor.visit_partial_head(PartialHead.__new__(PartialHead, indices, values))

    def contains_default_rule(self) -> bool:
        """
        Returns whether the model contains a default rule or not.

        :return: True, if the model contains a default rule, False otherwise
        """
        return self.rule_list_ptr.get().containsDefaultRule()

    def is_default_rule_taking_precedence(self) -> bool:
        """
        Returns whether the default rule takes precedence over the remaining rules or not.

        :return: True, if the default rule takes precedence over the remaining rules, False otherwise
        """
        return self.rule_list_ptr.get().isDefaultRuleTakingPrecedence()

    def visit(self, visitor: RuleModelVisitor):
        self.visitor = visitor
        self.rule_list_ptr.get().visit(
            wrapEmptyBodyVisitor(<void*>self, <EmptyBodyCythonVisitor>self.__visit_empty_body),
            wrapConjunctiveBodyVisitor(<void*>self, <ConjunctiveBodyCythonVisitor>self.__visit_conjunctive_body),
            wrapCompleteBinaryHeadVisitor(<void*>self,
                                          <CompleteBinaryHeadCythonVisitor>self.__visit_complete_binary_head),
            wrapComplete32BitHeadVisitor(<void*>self, <Complete32BitHeadCythonVisitor>self.__visit_complete_32bit_head),
            wrapComplete64BitHeadVisitor(<void*>self, <Complete64BitHeadCythonVisitor>self.__visit_complete_64bit_head),
            wrapPartialBinaryHeadVisitor(<void*>self, <PartialBinaryHeadCythonVisitor>self.__visit_partial_binary_head),
            wrapPartial32BitHeadVisitor(<void*>self, <Partial32BitHeadCythonVisitor>self.__visit_partial_32bit_head),
            wrapPartial64BitHeadVisitor(<void*>self, <Partial64BitHeadCythonVisitor>self.__visit_partial_64bit_head))
        self.visitor = None

    def visit_used(self, visitor: RuleModelVisitor):
        self.visitor = visitor
        self.rule_list_ptr.get().visitUsed(
            wrapEmptyBodyVisitor(<void*>self, <EmptyBodyCythonVisitor>self.__visit_empty_body),
            wrapConjunctiveBodyVisitor(<void*>self, <ConjunctiveBodyCythonVisitor>self.__visit_conjunctive_body),
            wrapCompleteBinaryHeadVisitor(<void*>self,
                                          <CompleteBinaryHeadCythonVisitor>self.__visit_complete_binary_head),
            wrapComplete32BitHeadVisitor(<void*>self, <Complete32BitHeadCythonVisitor>self.__visit_complete_32bit_head),
            wrapComplete64BitHeadVisitor(<void*>self, <Complete64BitHeadCythonVisitor>self.__visit_complete_64bit_head),
            wrapPartialBinaryHeadVisitor(<void*>self, <PartialBinaryHeadCythonVisitor>self.__visit_partial_binary_head),
            wrapPartial32BitHeadVisitor(<void*>self, <Partial32BitHeadCythonVisitor>self.__visit_partial_32bit_head),
            wrapPartial64BitHeadVisitor(<void*>self, <Partial64BitHeadCythonVisitor>self.__visit_partial_64bit_head))
        self.visitor = None

    def __reduce__(self):
        cdef bytes state = image_to_bytes(serializeRuleModel(dereference(self.rule_list_ptr.get())))
        return (RuleList, (), state)

    def __setstate__(self, state):
        check_pickled_state(state, 'RuleList')
        cdef const uint8[::1] data = state
        cdef const uint8* data_ptr = &data[0] if data.shape[0] > 0 else NULL
        cdef unique_ptr[IRuleModel] rule_model_ptr = deserializeRuleModel(data_ptr, data.shape[0])
        cdef IRuleList* rule_list_ptr = dynamic_cast[RuleListPtr](rule_model_ptr.get())

        if rule_list_ptr == NULL:
            raise ValueError('The pickled data does not contain an object of type RuleList')

        rule_model_ptr.release()
        self.rule_list_ptr = unique_ptr[IRuleList](rule_list_ptr)
