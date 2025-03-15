cimport numpy as npc

from libcpp cimport bool
from libcpp.cast cimport dynamic_cast
from libcpp.memory cimport unique_ptr

from mlrl.common.cython._types cimport float32, float64, int32, uint32


cdef extern from "mlrl/common/model/body.hpp" nogil:

    cdef cppclass IBody:
        pass


cdef extern from "mlrl/common/model/body_empty.hpp" nogil:

    cdef cppclass EmptyBodyImpl"EmptyBody"(IBody):
        pass


cdef extern from "mlrl/common/model/body_conjunctive.hpp" nogil:

    cdef cppclass ConjunctiveBodyImpl"ConjunctiveBody"(IBody):

        ctypedef float32* numerical_threshold_iterator

        ctypedef const float32* numerical_threshold_const_iterator

        ctypedef int32* ordinal_threshold_iterator

        ctypedef const int32* ordinal_threshold_const_iterator

        ctypedef int32* nominal_threshold_iterator

        ctypedef const int32* nominal_threshold_const_iterator

        ctypedef uint32* index_iterator

        ctypedef const uint32* index_const_iterator

        # Constructors:

        ConjunctiveBodyImpl(uint32 numNumericalLeq, uint32 numNumericalGr, uint32 numNominalEq, uint32 numNominalNeq)

        # Functions:

        uint32 getNumNumericalLeq() const

        numerical_threshold_iterator numerical_leq_thresholds_begin()

        numerical_threshold_const_iterator numerical_leq_thresholds_cbegin() const

        index_iterator numerical_leq_indices_begin()

        index_const_iterator numerical_leq_indices_cbegin() const

        uint32 getNumNumericalGr() const

        numerical_threshold_iterator numerical_gr_thresholds_begin()

        numerical_threshold_const_iterator numerical_gr_thresholds_cbegin() const

        index_iterator numerical_gr_indices_begin()

        index_const_iterator numerical_gr_indices_cbegin() const

        uint32 getNumOrdinalLeq() const

        ordinal_threshold_iterator ordinal_leq_thresholds_begin()

        ordinal_threshold_const_iterator ordinal_leq_thresholds_cbegin() const

        index_iterator ordinal_leq_indices_begin()

        index_const_iterator ordinal_leq_indices_cbegin() const

        uint32 getNumOrdinalGr() const

        ordinal_threshold_iterator ordinal_gr_thresholds_begin()

        ordinal_threshold_const_iterator ordinal_gr_thresholds_cbegin() const

        index_iterator ordinal_gr_indices_begin()

        index_const_iterator ordinal_gr_indices_cbegin() const

        uint32 getNumNominalEq() const

        nominal_threshold_iterator nominal_eq_thresholds_begin()

        nominal_threshold_const_iterator nominal_eq_thresholds_cbegin() const

        index_iterator nominal_eq_indices_begin()

        index_const_iterator nominal_eq_indices_cbegin() const

        uint32 getNumNominalNeq() const

        nominal_threshold_iterator nominal_neq_thresholds_begin()

        nominal_threshold_const_iterator nominal_neq_thresholds_cbegin() const

        index_iterator nominal_neq_indices_begin()

        index_const_iterator nominal_neq_indices_cbegin() const


cdef extern from "mlrl/common/model/head.hpp" nogil:

    cdef cppclass IHead:
        pass


cdef extern from "mlrl/common/model/head_complete.hpp" nogil:

    cdef cppclass Complete32BitHeadImpl"CompleteHead<float32>"(IHead):

        ctypedef float32* value_iterator

        ctypedef const float32* value_const_iterator

        # Functions:

        uint32 getNumElements() const

        value_iterator values_begin()

        value_const_iterator values_cbegin() const


    cdef cppclass Complete64BitHeadImpl"CompleteHead<float64>"(IHead):

        ctypedef float64* value_iterator

        ctypedef const float64* value_const_iterator

        # Functions:

        uint32 getNumElements() const

        value_iterator values_begin()

        value_const_iterator values_cbegin() const


cdef extern from "mlrl/common/model/head_partial.hpp" nogil:

    cdef cppclass Partial32BitHeadImpl"PartialHead<float32>"(IHead):

        ctypedef float32* value_iterator

        ctypedef const float32* value_const_iterator

        ctypedef uint32* index_iterator

        ctypedef const uint32* index_const_iterator

        # Functions:

        uint32 getNumElements() const

        value_iterator values_begin()

        value_const_iterator values_cbegin() const

        index_iterator indices_begin()

        index_const_iterator indices_cbegin() const


    cdef cppclass Partial64BitHeadImpl"PartialHead<float64>"(IHead):

        ctypedef float64* value_iterator

        ctypedef const float64* value_const_iterator

        ctypedef uint32* index_iterator

        ctypedef const uint32* index_const_iterator

        # Functions:

        uint32 getNumElements() const

        value_iterator values_begin()

        value_const_iterator values_cbegin() const

        index_iterator indices_begin()

        index_const_iterator indices_cbegin() const


ctypedef void (*EmptyBodyVisitor)(const EmptyBodyImpl&)

ctypedef void (*ConjunctiveBodyVisitor)(const ConjunctiveBodyImpl&)

ctypedef void (*Complete32BitHeadVisitor)(const Complete32BitHeadImpl&)

ctypedef void (*Complete64BitHeadVisitor)(const Complete64BitHeadImpl&)

ctypedef void (*Partial32BitHeadVisitor)(const Partial32BitHeadImpl&)

ctypedef void (*Partial64BitHeadVisitor)(const Partial64BitHeadImpl&)


cdef extern from "mlrl/common/model/rule_model.hpp" nogil:

    cdef cppclass IRuleModel:

        # Functions:

        uint32 getNumRules() const

        uint32 getNumUsedRules() const

        void setNumUsedRules(uint32 numUsedRules)


cdef extern from "mlrl/common/model/rule_list.hpp" nogil:

    cdef cppclass IRuleList(IRuleModel):

        # Functions:

        void addDefaultRule(unique_ptr[IHead] headPtr)

        void addRule(unique_ptr[IBody] bodyPtr, unique_ptr[IHead] headPtr)

        bool containsDefaultRule() const

        bool isDefaultRuleTakingPrecedence() const

        void visit(EmptyBodyVisitor emptyBodyVisitor, ConjunctiveBodyVisitor conjunctiveBodyVisitor,
                   Complete32BitHeadVisitor complete32BitHeadVisitor, Complete64BitHeadVisitor complete64BitHeadVisitor,
                   Partial32BitHeadVisitor partial32BITHeadVisitor,
                   Partial64BitHeadVisitor partial64BITHeadVisitor) const

        void visitUsed(EmptyBodyVisitor emptyBodyVisitor, ConjunctiveBodyVisitor conjunctiveBodyVisitor,
                       Complete32BitHeadVisitor complete32BitHeadVisitor,
                       Complete64BitHeadVisitor complete64BitHeadVisitor,
                       Partial32BitHeadVisitor partial32BitHeadVisitor,
                       Partial64BitHeadVisitor partial64BitHeadVisitor) const


    unique_ptr[IRuleList] createRuleList(bool defaultRuleTakesPrecedence)


ctypedef IRuleList* RuleListPtr


cdef extern from *:
    """
    #include "mlrl/common/model/body.hpp"
    #include "mlrl/common/model/head.hpp"


    typedef void (*EmptyBodyCythonVisitor)(void*, const EmptyBody&);

    typedef void (*ConjunctiveBodyCythonVisitor)(void*, const ConjunctiveBody&);

    typedef void (*Complete32BitHeadCythonVisitor)(void*, const CompleteHead<float32>&);

    typedef void (*Complete64BitHeadCythonVisitor)(void*, const CompleteHead<float64>&);

    typedef void (*Partial32BitHeadCythonVisitor)(void*, const PartialHead<float32>&);

    typedef void (*Partial64BitHeadCythonVisitor)(void*, const PartialHead<float64>&);

    static inline IBody::EmptyBodyVisitor wrapEmptyBodyVisitor(void* self, EmptyBodyCythonVisitor visitor) {
        return [=](const EmptyBody& body) {
            visitor(self, body);
        };
    }

    static inline IBody::ConjunctiveBodyVisitor wrapConjunctiveBodyVisitor(void* self,
                                                                           ConjunctiveBodyCythonVisitor visitor) {
        return [=](const ConjunctiveBody& body) {
            visitor(self, body);
        };
    }

    static inline IHead::CompleteHeadVisitor<float32> wrapComplete32BitHeadVisitor(
      void* self, Complete32BitHeadCythonVisitor visitor) {
        return [=](const CompleteHead<float32>& head) {
            visitor(self, head);
        };
    }

    static inline IHead::CompleteHeadVisitor<float64> wrapComplete64BitHeadVisitor(
      void* self, Complete64BitHeadCythonVisitor visitor) {
        return [=](const CompleteHead<float64>& head) {
            visitor(self, head);
        };
    }

    static inline IHead::PartialHeadVisitor<float32> wrapPartial32BitHeadVisitor(
      void* self, Partial32BitHeadCythonVisitor visitor) {
        return [=](const PartialHead<float32>& head) {
            visitor(self, head);
        };
    }

    static inline IHead::PartialHeadVisitor<float64> wrapPartial64BitHeadVisitor(
      void* self, Partial64BitHeadCythonVisitor visitor) {
        return [=](const PartialHead<float64>& head) {
            visitor(self, head);
        };
    }
    """

    ctypedef void (*EmptyBodyCythonVisitor)(void*, const EmptyBodyImpl&)

    ctypedef void (*ConjunctiveBodyCythonVisitor)(void*, const ConjunctiveBodyImpl&)

    ctypedef void (*Complete32BitHeadCythonVisitor)(void*, const Complete32BitHeadImpl&)

    ctypedef void (*Complete64BitHeadCythonVisitor)(void*, const Complete64BitHeadImpl&)

    ctypedef void (*Partial32BitHeadCythonVisitor)(void*, const Partial32BitHeadImpl&)

    ctypedef void (*Partial64BitHeadCythonVisitor)(void*, const Partial64BitHeadImpl&)

    EmptyBodyVisitor wrapEmptyBodyVisitor(void* self, EmptyBodyCythonVisitor visitor)

    ConjunctiveBodyVisitor wrapConjunctiveBodyVisitor(void* self, ConjunctiveBodyCythonVisitor visitor)

    Complete32BitHeadVisitor wrapComplete32BitHeadVisitor(void* self, Complete32BitHeadCythonVisitor visitor)

    Complete64BitHeadVisitor wrapComplete64BitHeadVisitor(void* self, Complete64BitHeadCythonVisitor visitor)

    Partial32BitHeadVisitor wrapPartial32BitHeadVisitor(void* self, Partial32BitHeadCythonVisitor visitor)

    Partial64BitHeadVisitor wrapPartial64BitHeadVisitor(void* self, Partial64BitHeadCythonVisitor visitor)


cdef class EmptyBody:
    pass


cdef class ConjunctiveBody:

    # Attributes:

    cdef readonly npc.ndarray numerical_leq_indices

    cdef readonly npc.ndarray numerical_leq_thresholds

    cdef readonly npc.ndarray numerical_gr_indices

    cdef readonly npc.ndarray numerical_gr_thresholds

    cdef readonly npc.ndarray ordinal_leq_indices

    cdef readonly npc.ndarray ordinal_leq_thresholds

    cdef readonly npc.ndarray ordinal_gr_indices

    cdef readonly npc.ndarray ordinal_gr_thresholds

    cdef readonly npc.ndarray nominal_eq_indices

    cdef readonly npc.ndarray nominal_eq_thresholds

    cdef readonly npc.ndarray nominal_neq_indices

    cdef readonly npc.ndarray nominal_neq_thresholds


cdef class CompleteHead:

    # Attributes:

    cdef readonly npc.ndarray scores


cdef class PartialHead:

    # Attributes:

    cdef readonly npc.ndarray indices

    cdef readonly npc.ndarray scores


cdef class RuleModel:

    # Functions:

    cdef IRuleModel* get_rule_model_ptr(self)


cdef class RuleList(RuleModel):

    # Attributes:

    cdef unique_ptr[IRuleList] rule_list_ptr

    cdef object visitor

    cdef object state

    # Functions:

    cdef __visit_empty_body(self, const EmptyBodyImpl& body)

    cdef __visit_conjunctive_body(self, const ConjunctiveBodyImpl& body)

    cdef __visit_complete_32bit_head(self, const Complete32BitHeadImpl& head)

    cdef __visit_complete_64bit_head(self, const Complete64BitHeadImpl& head)

    cdef __visit_partial_32bit_head(self, const Partial32BitHeadImpl& head)

    cdef __visit_partial_64bit_head(self, const Partial64BitHeadImpl& head)

    cdef __serialize_empty_body(self, const EmptyBodyImpl& body)

    cdef __serialize_conjunctive_body(self, const ConjunctiveBodyImpl& body)

    cdef __serialize_complete_32bit_head(self, const Complete32BitHeadImpl& head)

    cdef __serialize_complete_64bit_head(self, const Complete64BitHeadImpl& head)

    cdef __serialize_partial_32bit_head(self, const Partial32BitHeadImpl& head)

    cdef __serialize_partial_64bit_head(self, const Partial64BitHeadImpl& head)

    cdef unique_ptr[IBody] __deserialize_body(self, object body_state)

    cdef unique_ptr[IBody] __deserialize_conjunctive_body(self, object body_state)

    cdef unique_ptr[IHead] __deserialize_head(self, object head_state)

    cdef unique_ptr[IHead] __deserialize_complete_32bit_head(self, object head_state)

    cdef unique_ptr[IHead] __deserialize_complete_64bit_head(self, object head_state)

    cdef unique_ptr[IHead] __deserialize_partial_32bit_head(self, object head_state)

    cdef unique_ptr[IHead] __deserialize_partial_64bit_head(self, object head_state)


cdef inline RuleModel create_rule_model(unique_ptr[IRuleModel] rule_model_ptr):
    cdef IRuleModel* ptr = rule_model_ptr.release()
    cdef IRuleList* rule_list_ptr = dynamic_cast[RuleListPtr](ptr)
    cdef RuleList rule_list

    if rule_list_ptr != NULL:
        rule_list = RuleList.__new__(RuleList)
        rule_list.rule_list_ptr = unique_ptr[IRuleList](rule_list_ptr)
        return rule_list
    else:
        del ptr
        raise RuntimeError('Encountered unsupported IRuleModel object')
