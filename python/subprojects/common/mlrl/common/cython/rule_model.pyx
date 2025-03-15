"""
@author Michael Rapp (michael.rapp.ml@gmail.com)
"""
from libcpp.memory cimport make_unique
from libcpp.utility cimport move

from abc import abstractmethod

import numpy as np

SERIALIZATION_VERSION = 4


cdef class EmptyBody:
    """
    A body of a rule that does not contain any conditions.
    """
    pass


cdef class ConjunctiveBody:
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


cdef class CompleteHead:
    """
    A head of a rule that predicts numerical scores for all available outputs.
    """

    def __cinit__(self, npc.ndarray scores not None):
        """
        :param scores: A `npc.ndarray`, shape `(num_predictions)` that stores the predicted scores
        """
        self.scores = scores


cdef class PartialHead:
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


cdef class Partial64BitHead:
    """
    A head of a rule that predicts numerical scores, represented by 64-bit floating point values, for a subset of the
    available outputs.
    """

    def __cinit__(self, const uint32[::1] indices not None, const float64[::1] scores not None):
        """
        :param indices: A contiguous array of type `uint32`, shape `(num_predictions)` that stores the output indices
        :param scores:  A contiguous array of type `float64`, shape `(num_predictions)` that stores the predicted scores
        """
        self.indices = np.asarray(indices)
        self.scores = np.asarray(scores)


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
        pass

    @abstractmethod
    def visit_conjunctive_body(self, body: ConjunctiveBody):
        """
        Must be implemented by subclasses in order to visit the bodies of rule that are given as a conjunction of
        several conditions.

        :param body: A `ConjunctiveBody` to be visited
        """
        pass

    @abstractmethod
    def visit_complete_head(self, head: CompleteHead):
        """
        Must be implemented by subclasses in order to visit the heads of rules that predict numerical scores for all
        available outputs.

        :param head: A `CompleteHead` to be visited
        """
        pass
    
    @abstractmethod
    def visit_partial_head(self, head: PartialHead):
        """
        Must be implemented by subclasses in order to visit the heads of rules that predict numerical scores for a
        subset of the available outputs.

        :param head: A `PartialHead` to be visited
        """
        pass


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

    def visit(self, visitor: RuleModelVisitor):
        """
        Visits the bodies and heads of all rules that are contained in this model, including the default rule, if
        available.

        :param visitor: The `RuleModelVisitor` that should be used to access the bodies and heads
        """
        pass

    def visit_used(self, visitor: RuleModelVisitor):
        """
        Visits the bodies and heads of all used rules that are contained in this model, including the default rule, if
        available.

        :param visitor: The `RuleModelVisitor` that should be used to access the bodies and heads
        """
        pass


cdef class RuleList(RuleModel):
    """
    A rule-based model that stores several rules in an ordered list.
    """

    def __cinit__(self):
        self.visitor = None
        self.state = None

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

    cdef __visit_complete_32bit_head(self, const Complete32BitHeadImpl& head):
        cdef uint32 num_elements = head.getNumElements()
        cdef npc.ndarray values = np.asarray(<float32[:num_elements]>head.values_cbegin())
        self.visitor.visit_complete_head(CompleteHead.__new__(CompleteHead, values))
    
    cdef __visit_complete_64bit_head(self, const Complete64BitHeadImpl& head):
        cdef uint32 num_elements = head.getNumElements()
        cdef npc.ndarray values = np.asarray(<float64[:num_elements]>head.values_cbegin())
        self.visitor.visit_complete_head(CompleteHead.__new__(CompleteHead, values))
    
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

    cdef __serialize_empty_body(self, const EmptyBodyImpl& body):
        cdef object body_state = None
        cdef object rule_state = [body_state, None]
        self.state.append(rule_state)

    cdef __serialize_conjunctive_body(self, const ConjunctiveBodyImpl& body):
        cdef uint32 num_numerical_leq = body.getNumNumericalLeq()
        cdef uint32 num_numerical_gr = body.getNumNumericalGr()
        cdef uint32 num_ordinal_leq = body.getNumOrdinalLeq()
        cdef uint32 num_ordinal_gr = body.getNumOrdinalGr()
        cdef uint32 num_nominal_eq = body.getNumNominalEq()
        cdef uint32 num_nominal_neq = body.getNumNominalNeq()
        cdef object body_state = (
            np.asarray(<float32[:num_numerical_leq]>body.numerical_leq_thresholds_cbegin()) \
                if num_numerical_leq > 0 else None,
            np.asarray(<uint32[:num_numerical_leq]>body.numerical_leq_indices_cbegin()) \
                if num_numerical_leq > 0 else None,
            np.asarray(<float32[:num_numerical_gr]>body.numerical_gr_thresholds_cbegin()) \
                if num_numerical_gr > 0 else None,
            np.asarray(<uint32[:num_numerical_gr]>body.numerical_gr_indices_cbegin()) \
                if num_numerical_gr > 0 else None,
            np.asarray(<int32[:num_ordinal_leq]>body.ordinal_leq_thresholds_cbegin()) \
                if num_ordinal_leq > 0 else None,
            np.asarray(<uint32[:num_ordinal_leq]>body.ordinal_leq_indices_cbegin()) \
                if num_ordinal_leq > 0 else None,
            np.asarray(<int32[:num_ordinal_gr]>body.ordinal_gr_thresholds_cbegin()) \
                if num_ordinal_gr > 0 else None,
            np.asarray(<uint32[:num_ordinal_gr]>body.ordinal_gr_indices_cbegin()) \
                if num_ordinal_gr > 0 else None,
            np.asarray(<int32[:num_nominal_eq]>body.nominal_eq_thresholds_cbegin()) \
                if num_nominal_eq > 0 else None,
            np.asarray(<uint32[:num_nominal_eq]>body.nominal_eq_indices_cbegin()) \
                if num_nominal_eq > 0 else None,
            np.asarray(<int32[:num_nominal_neq]>body.nominal_neq_thresholds_cbegin()) \
                if num_nominal_neq > 0 else None,
            np.asarray(<uint32[:num_nominal_neq]>body.nominal_neq_indices_cbegin()) \
                if num_nominal_neq > 0 else None,
        )
        cdef object rule_state = [body_state, None]
        self.state.append(rule_state)

    cdef __serialize_complete_32bit_head(self, const Complete32BitHeadImpl& head):
        cdef uint32 num_elements = head.getNumElements()
        cdef object head_state = (np.asarray(<float32[:num_elements]>head.values_cbegin()),)
        cdef object rule_state = self.state[len(self.state) - 1]
        rule_state[1] = head_state
    
    cdef __serialize_complete_64bit_head(self, const Complete64BitHeadImpl& head):
        cdef uint32 num_elements = head.getNumElements()
        cdef object head_state = (np.asarray(<float64[:num_elements]>head.values_cbegin()),)
        cdef object rule_state = self.state[len(self.state) - 1]
        rule_state[1] = head_state

    cdef __serialize_partial_32bit_head(self, const Partial32BitHeadImpl& head):
        cdef uint32 num_elements = head.getNumElements()
        cdef object head_state = (np.asarray(<float32[:num_elements]>head.values_cbegin()),
                                  np.asarray(<uint32[:num_elements]>head.indices_cbegin()))
        cdef object rule_state = self.state[len(self.state) - 1]
        rule_state[1] = head_state
    
    cdef __serialize_partial_64bit_head(self, const Partial64BitHeadImpl& head):
        cdef uint32 num_elements = head.getNumElements()
        cdef object head_state = (np.asarray(<float64[:num_elements]>head.values_cbegin()),
                                  np.asarray(<uint32[:num_elements]>head.indices_cbegin()))
        cdef object rule_state = self.state[len(self.state) - 1]
        rule_state[1] = head_state

    cdef unique_ptr[IBody] __deserialize_body(self, object body_state):
        cdef unique_ptr[IBody] body_ptr

        if body_state is not None:
            body_ptr = move(self.__deserialize_conjunctive_body(body_state))

        return move(body_ptr)

    cdef unique_ptr[IBody] __deserialize_conjunctive_body(self, object body_state):
        cdef const float32[::1] numerical_leq_thresholds = body_state[0]
        cdef const uint32[::1] numerical_leq_indices = body_state[1]
        cdef const float32[::1] numerical_gr_thresholds = body_state[2]
        cdef const uint32[::1] numerical_gr_indices = body_state[3]
        cdef const int32[::1] ordinal_leq_thresholds = body_state[4]
        cdef const uint32[::1] ordinal_leq_indices = body_state[5]
        cdef const int32[::1] ordinal_gr_thresholds = body_state[6]
        cdef const uint32[::1] ordinal_gr_indices = body_state[7]
        cdef const int32[::1] nominal_eq_thresholds = body_state[8]
        cdef const uint32[::1] nominal_eq_indices = body_state[9]
        cdef const int32[::1] nominal_neq_thresholds = body_state[10]
        cdef const uint32[::1] nominal_neq_indices = body_state[11]
        cdef uint32 num_numerical_leq = numerical_leq_thresholds.shape[0] if numerical_leq_thresholds is not None else 0
        cdef uint32 num_numerical_gr = numerical_gr_thresholds.shape[0] if numerical_gr_thresholds is not None else 0
        cdef uint32 num_ordinal_leq = ordinal_leq_thresholds.shape[0] if ordinal_leq_thresholds is not None else 0
        cdef uint32 num_ordinal_gr = ordinal_gr_thresholds.shape[0] if ordinal_gr_thresholds is not None else 0
        cdef uint32 num_nominal_eq = nominal_eq_thresholds.shape[0] if nominal_eq_thresholds is not None else 0
        cdef uint32 num_nominal_neq = nominal_neq_thresholds.shape[0] if nominal_neq_thresholds is not None else 0
        cdef unique_ptr[ConjunctiveBodyImpl] body_ptr = make_unique[ConjunctiveBodyImpl](
            num_numerical_leq, num_numerical_gr, num_ordinal_leq, num_ordinal_gr, num_nominal_eq, num_nominal_neq)
        cdef ConjunctiveBodyImpl.numerical_threshold_iterator numerical_threshold_iterator = \
            body_ptr.get().numerical_leq_thresholds_begin()
        cdef ConjunctiveBodyImpl.index_iterator index_iterator = body_ptr.get().numerical_leq_indices_begin()
        cdef uint32 i

        for i in range(num_numerical_leq):
            numerical_threshold_iterator[i] = numerical_leq_thresholds[i]
            index_iterator[i] = numerical_leq_indices[i]

        numerical_threshold_iterator = body_ptr.get().numerical_gr_thresholds_begin()
        index_iterator = body_ptr.get().numerical_gr_indices_begin()

        for i in range(num_numerical_gr):
            numerical_threshold_iterator[i] = numerical_gr_thresholds[i]
            index_iterator[i] = numerical_gr_indices[i]

        cdef ConjunctiveBodyImpl.ordinal_threshold_iterator ordinal_threshold_iterator = \
            body_ptr.get().ordinal_leq_thresholds_begin()
        index_iterator = body_ptr.get().ordinal_leq_indices_begin()

        for i in range(num_ordinal_leq):
            ordinal_threshold_iterator[i] = ordinal_leq_thresholds[i]
            index_iterator[i] = ordinal_leq_indices[i]

        ordinal_threshold_iterator = body_ptr.get().ordinal_gr_thresholds_begin()
        index_iterator = body_ptr.get().ordinal_gr_indices_begin()

        for i in range(num_ordinal_gr):
            ordinal_threshold_iterator[i] = ordinal_gr_thresholds[i]
            index_iterator[i] = ordinal_gr_indices[i]

        cdef ConjunctiveBodyImpl.nominal_threshold_iterator nominal_threshold_iterator = \
            body_ptr.get().nominal_eq_thresholds_begin()
        index_iterator = body_ptr.get().nominal_eq_indices_begin()

        for i in range(num_nominal_eq):
            nominal_threshold_iterator[i] = nominal_eq_thresholds[i]
            index_iterator[i] = nominal_eq_indices[i]

        nominal_threshold_iterator = body_ptr.get().nominal_neq_thresholds_begin()
        index_iterator = body_ptr.get().nominal_neq_indices_begin()

        for i in range(num_nominal_neq):
            nominal_threshold_iterator[i] = nominal_neq_thresholds[i]
            index_iterator[i] = nominal_neq_indices[i]

        return <unique_ptr[IBody]>move(body_ptr)

    cdef unique_ptr[IHead] __deserialize_head(self, object head_state):
        if len(head_state) > 1:
            if head_state[0].dtype == np.float32:
                return move(self.__deserialize_partial_32bit_head(head_state))
            else:
                return move(self.__deserialize_partial_64bit_head(head_state))
        else:
            if head_state[0].dtype == np.float32:
                return move(self.__deserialize_complete_32bit_head(head_state))
            else:
                return move(self.__deserialize_complete_64bit_head(head_state))

    cdef unique_ptr[IHead] __deserialize_complete_32bit_head(self, object head_state):
        cdef const float32[::1] scores = head_state[0]
        cdef uint32 num_elements = scores.shape[0]
        cdef unique_ptr[Complete32BitHeadImpl] head_ptr = make_unique[Complete32BitHeadImpl](num_elements)
        cdef Complete32BitHeadImpl.value_iterator value_iterator = head_ptr.get().values_begin()
        cdef uint32 i

        for i in range(num_elements):
            value_iterator[i] = scores[i]

        return <unique_ptr[IHead]>move(head_ptr)
    
    cdef unique_ptr[IHead] __deserialize_complete_64bit_head(self, object head_state):
        cdef const float64[::1] scores = head_state[0]
        cdef uint32 num_elements = scores.shape[0]
        cdef unique_ptr[Complete64BitHeadImpl] head_ptr = make_unique[Complete64BitHeadImpl](num_elements)
        cdef Complete64BitHeadImpl.value_iterator value_iterator = head_ptr.get().values_begin()
        cdef uint32 i

        for i in range(num_elements):
            value_iterator[i] = scores[i]

        return <unique_ptr[IHead]>move(head_ptr)

    cdef unique_ptr[IHead] __deserialize_partial_64bit_head(self, object head_state):
        cdef const float64[::1] scores = head_state[0]
        cdef const uint32[::1] indices = head_state[1]
        cdef uint32 num_elements = scores.shape[0]
        cdef unique_ptr[Partial64BitHeadImpl] head_ptr = make_unique[Partial64BitHeadImpl](num_elements)
        cdef Partial64BitHeadImpl.value_iterator value_iterator = head_ptr.get().values_begin()
        cdef Partial64BitHeadImpl.index_iterator index_iterator = head_ptr.get().indices_begin()
        cdef uint32 i

        for i in range(num_elements):
            value_iterator[i] = scores[i]
            index_iterator[i] = indices[i]

        return <unique_ptr[IHead]>move(head_ptr)

    cdef unique_ptr[IHead] __deserialize_partial_32bit_head(self, object head_state):
        cdef const float32[::1] scores = head_state[0]
        cdef const uint32[::1] indices = head_state[1]
        cdef uint32 num_elements = scores.shape[0]
        cdef unique_ptr[Partial32BitHeadImpl] head_ptr = make_unique[Partial32BitHeadImpl](num_elements)
        cdef Partial32BitHeadImpl.value_iterator value_iterator = head_ptr.get().values_begin()
        cdef Partial32BitHeadImpl.index_iterator index_iterator = head_ptr.get().indices_begin()
        cdef uint32 i

        for i in range(num_elements):
            value_iterator[i] = scores[i]
            index_iterator[i] = indices[i]

        return <unique_ptr[IHead]>move(head_ptr)

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
            wrapComplete32BitHeadVisitor(<void*>self, <Complete32BitHeadCythonVisitor>self.__visit_complete_32bit_head),
            wrapComplete64BitHeadVisitor(<void*>self, <Complete64BitHeadCythonVisitor>self.__visit_complete_64bit_head),
            wrapPartial32BitHeadVisitor(<void*>self, <Partial32BitHeadCythonVisitor>self.__visit_partial_32bit_head),
            wrapPartial64BitHeadVisitor(<void*>self, <Partial64BitHeadCythonVisitor>self.__visit_partial_64bit_head))
        self.visitor = None

    def visit_used(self, visitor: RuleModelVisitor):
        self.visitor = visitor
        self.rule_list_ptr.get().visitUsed(
            wrapEmptyBodyVisitor(<void*>self, <EmptyBodyCythonVisitor>self.__visit_empty_body),
            wrapConjunctiveBodyVisitor(<void*>self, <ConjunctiveBodyCythonVisitor>self.__visit_conjunctive_body),
            wrapComplete32BitHeadVisitor(<void*>self, <Complete32BitHeadCythonVisitor>self.__visit_complete_32bit_head),
            wrapComplete64BitHeadVisitor(<void*>self, <Complete64BitHeadCythonVisitor>self.__visit_complete_64bit_head),
            wrapPartial32BitHeadVisitor(<void*>self, <Partial32BitHeadCythonVisitor>self.__visit_partial_32bit_head),
            wrapPartial64BitHeadVisitor(<void*>self, <Partial64BitHeadCythonVisitor>self.__visit_partial_64bit_head))
        self.visitor = None

    def __reduce__(self):
        self.state = []
        self.rule_list_ptr.get().visit(
            wrapEmptyBodyVisitor(<void*>self, <EmptyBodyCythonVisitor>self.__serialize_empty_body),
            wrapConjunctiveBodyVisitor(<void*>self, <ConjunctiveBodyCythonVisitor>self.__serialize_conjunctive_body),
            wrapComplete32BitHeadVisitor(<void*>self,
                                         <Complete32BitHeadCythonVisitor>self.__serialize_complete_32bit_head),
            wrapComplete64BitHeadVisitor(<void*>self,
                                         <Complete64BitHeadCythonVisitor>self.__serialize_complete_64bit_head),
            wrapPartial32BitHeadVisitor(<void*>self,
                                        <Partial32BitHeadCythonVisitor>self.__serialize_partial_32bit_head),
            wrapPartial64BitHeadVisitor(<void*>self,
                                        <Partial64BitHeadCythonVisitor>self.__serialize_partial_64bit_head))
        cdef bint default_rule_takes_precedence = self.rule_list_ptr.get().isDefaultRuleTakingPrecedence()
        cdef uint32 num_used_rules = self.rule_list_ptr.get().getNumUsedRules()
        cdef object state = (SERIALIZATION_VERSION, (self.state, default_rule_takes_precedence, num_used_rules))
        self.state = None
        return (RuleList, (), state)

    def __setstate__(self, state):
        cdef int version = state[0]

        if version != SERIALIZATION_VERSION:
            raise AssertionError('Version of the serialized RuleModel is ' + str(version) + ', expected '
                                 + str(SERIALIZATION_VERSION))

        cdef object model_state = state[1]
        cdef list rule_list = model_state[0]
        cdef bint default_rule_takes_precedence = model_state[1]
        cdef uint32 num_rules = len(rule_list)
        cdef unique_ptr[IRuleList] rule_list_ptr = createRuleList(default_rule_takes_precedence)
        cdef object rule_state
        cdef unique_ptr[IBody] body_ptr
        cdef unique_ptr[IHead] head_ptr
        cdef uint32 i

        for i in range(num_rules):
            rule_state = rule_list[i]
            body_ptr = self.__deserialize_body(rule_state[0])
            head_ptr = self.__deserialize_head(rule_state[1])

            if body_ptr.get() == NULL:
                rule_list_ptr.get().addDefaultRule(move(head_ptr))
            else:
                rule_list_ptr.get().addRule(move(body_ptr), move(head_ptr))

        cdef uint32 num_used_rules = model_state[2]
        rule_list_ptr.get().setNumUsedRules(num_used_rules)
        self.rule_list_ptr = move(rule_list_ptr)
