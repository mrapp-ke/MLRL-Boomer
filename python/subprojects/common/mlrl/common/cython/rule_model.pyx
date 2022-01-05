"""
@author Michael Rapp (michael.rapp.ml@gmail.com)
"""
from abc import abstractmethod

import numpy as np


cdef class EmptyBody:
    """
    A body of a rule that does not contain any conditions.
    """
    pass


cdef class ConjunctiveBody:
    """
    A body of a rule that is given as a conjunction of several conditions.
    """

    def __cinit__(self, const uint32[::1] leq_indices, const float32[::1] leq_thresholds, const uint32[::1] gr_indices,
                  const float32[::1] gr_thresholds, const uint32[::1] eq_indices, const float32[::1] eq_thresholds,
                  const uint32[::1] neq_indices, const float32[::1] neq_thresholds):
        """
        :param leq_indices:     A contiguous array of type `uint32`, shape `(num_leq_conditions)`, that stores the
                                feature indices of the conditions that use the <= operator or None, if no such
                                conditions are available
        :param leq_thresholds:  A contiguous array of type `float32`, shape `(num_leq_conditions)` that stores the
                                thresholds of the conditions that use the <= operator or None, if no such conditions are
                                available
        :param gr_indices:      A contiguous array of type `uint32`, shape `(num_gr_conditions)`, that stores the
                                feature indices of the conditions that use the > operator or None, if no such conditions
                                are available
        :param gr_thresholds:   A contiguous array of type `float32`, shape `(num_gr_conditions)` that stores the
                                thresholds of the conditions that use the > operator or None, if no such conditions are
                                available
        :param eq_indices:      A contiguous array of type `uint32`, shape `(num_eq_conditions)`, that stores the
                                feature indices of the conditions that use the == operator or None, if no such
                                conditions are available
        :param eq_thresholds:   A contiguous array of type `float32`, shape `(num_eq_conditions)` that stores the
                                thresholds of the conditions that use the == operator or None, if no such conditions are
                                available
        :param neq_indices:     A contiguous array of type `uint32`, shape `(num_neq_conditions)`, that stores the
                                feature indices of the conditions that use the != operator or None, if no such
                                conditions are available
        :param neq_thresholds:  A contiguous array of type `float32`, shape `(num_neq_conditions)` that stores the
                                thresholds of the conditions that use the != operator or None, if no such conditions are
                                available
        """
        self.leq_indices = np.asarray(leq_indices) if leq_indices is not None else None
        self.leq_thresholds = np.asarray(leq_thresholds) if leq_thresholds is not None else None
        self.gr_indices = np.asarray(gr_indices) if gr_indices is not None else None
        self.gr_thresholds = np.asarray(gr_thresholds) if gr_thresholds is not None else None
        self.eq_indices = np.asarray(eq_indices) if eq_indices is not None else None
        self.eq_thresholds = np.asarray(eq_thresholds) if eq_thresholds is not None else None
        self.neq_indices = np.asarray(neq_indices) if neq_indices is not None else None
        self.neq_thresholds = np.asarray(neq_thresholds) if neq_thresholds is not None else None


cdef class CompleteHead:
    """
    A head of a rule that predicts for all available labels.
    """

    def __cinit__(self, const float64[::1] scores not None):
        """
        :param scores: A contiguous array of type `float64`, shape `(num_predictions)` that stores the predicted scores
        """
        self.scores = np.asarray(scores)


cdef class PartialHead:
    """
    A head of a rule that predicts for a subset of the available labels.
    """

    def __cinit__(self, const uint32[::1] indices not None, const float64[::1] scores not None):
        """
        :param indices: A contiguous array of type `uint32`, shape `(num_predictions)` that stores the label indices
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
        Must be implemented by subclasses in order to visit the heads of rules that predict for all available labels.

        :param head: A `CompleteHead` to be visited
        """
        pass

    @abstractmethod
    def visit_partial_head(self, head: PartialHead):
        """
        Must be implemented by subclasses in order to visit the heads of rules that predict for a subset of the
        available labels.

        :param head: A `PartialHead` to be visited
        """
        pass


cdef class RuleModel:
    """
    A wrapper for the pure virtual C++ class `IRuleModel`.
    """

    cdef IRuleModel* get_rule_model_ptr(self):
        pass

    def get_num_rules(self) -> int:
        """
        Returns the total number of rules in the model.

        :return The total number of rules in the model
        """
        return self.get_rule_model_ptr().getNumRules()

    def get_num_used_rules(self) -> int:
        """
        Returns the number of used rules in the model.

        :return The number of used rules in the model
        """
        return self.get_rule_model_ptr().getNumUsedRules()

    def set_num_used_rules(self, num_used_rules: int):
        """
        Sets the number of used rules in the model.

        :param num_used_rules: The number of used rules to be set
        """
        self.get_rule_model_ptr().setNumUsedRules(num_used_rules)

    def visit(self, visitor: RuleModelVisitor):
        """
        Visits the bodies and heads of the rules in the model.

        :param visitor: The `RuleModelVisitor` that should be used to access the bodies and heads
        """
        pass


cdef class RuleList(RuleModel):
    """
    A wrapper for the pure virtual C++ class `IRuleList`.
    """

    def __cinit__(self):
        self.visitor = None

    cdef IRuleModel* get_rule_model_ptr(self):
        return self.rule_list_ptr.get()

    cdef __visit_empty_body(self, const EmptyBodyImpl& body):
        self.visitor.visit_empty_body(EmptyBody.__new__(EmptyBody))

    cdef __visit_conjunctive_body(self, const ConjunctiveBodyImpl& body):
        cdef uint32 num_leq = body.getNumLeq()
        cdef const uint32[::1] leq_indices = <uint32[:num_leq]>body.leq_indices_cbegin() if num_leq > 0 else None
        cdef const float32[::1] leq_thresholds = <float32[:num_leq]>body.leq_thresholds_cbegin() if num_leq > 0 else None
        cdef uint32 num_gr = body.getNumGr()
        cdef const uint32[::1] gr_indices = <uint32[:num_gr]>body.gr_indices_cbegin() if num_gr > 0 else None
        cdef const float32[::1] gr_thresholds = <float32[:num_gr]>body.gr_thresholds_cbegin() if num_gr > 0 else None
        cdef uint32 num_eq = body.getNumEq()
        cdef const uint32[::1] eq_indices = <uint32[:num_eq]>body.eq_indices_cbegin() if num_eq > 0 else None
        cdef const float32[::1] eq_thresholds = <float32[:num_eq]>body.eq_thresholds_cbegin() if num_eq > 0 else None
        cdef uint32 num_neq = body.getNumNeq()
        cdef const uint32[::1] neq_indices = <uint32[:num_neq]>body.neq_indices_cbegin() if num_neq > 0 else None
        cdef const float32[::1] neq_thresholds = <float32[:num_neq]>body.neq_thresholds_cbegin() if num_neq > 0 else None
        self.visitor.visit_conjunctive_body(ConjunctiveBody.__new__(ConjunctiveBody, leq_indices, leq_thresholds,
                                                                    gr_indices, gr_thresholds, eq_indices,
                                                                    eq_thresholds, neq_indices, neq_thresholds))

    cdef __visit_complete_head(self, const CompleteHeadImpl& head):
        cdef uint32 num_elements = head.getNumElements()
        cdef const float64[::1] scores = <float64[:num_elements]>head.scores_cbegin()
        self.visitor.visit_complete_head(CompleteHead.__new__(CompleteHead, scores))

    cdef __visit_partial_head(self, const PartialHeadImpl& head):
        cdef uint32 num_elements = head.getNumElements()
        cdef const uint32[::1] indices = <uint32[:num_elements]>head.indices_cbegin()
        cdef const float64[::1] scores = <float64[:num_elements]>head.scores_cbegin()
        self.visitor.visit_partial_head(PartialHead.__new__(PartialHead, indices, scores))

    def visit(self, visitor: RuleModelVisitor):
        self.visitor = visitor
        self.rule_list_ptr.get().visitUsed(
            wrapEmptyBodyVisitor(<void*>self, <EmptyBodyCythonVisitor>self.__visit_empty_body),
            wrapConjunctiveBodyVisitor(<void*>self, <ConjunctiveBodyCythonVisitor>self.__visit_conjunctive_body),
            wrapCompleteHeadVisitor(<void*>self, <CompleteHeadCythonVisitor>self.__visit_complete_head),
            wrapPartialHeadVisitor(<void*>self, <PartialHeadCythonVisitor>self.__visit_partial_head))
        self.visitor = None
