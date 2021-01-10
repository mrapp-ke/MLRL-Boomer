from boomer.common._types cimport uint32, intp, float32, float64
from boomer.common.head_refinement cimport AbstractPrediction

from libcpp cimport bool
from libcpp.list cimport list as double_linked_list
from libcpp.memory cimport unique_ptr, shared_ptr


cdef extern from "cpp/model/condition.h" nogil:

    cdef enum Comparator:
        LEQ
        GR
        EQ
        NEQ


    cdef struct Condition:
        uint32 featureIndex
        Comparator comparator
        float32 threshold
        intp start
        intp end
        bool covered
        uint32 coveredWeights


cdef extern from "cpp/model/condition_list.h" nogil:

    cdef cppclass ConditionList:

        ctypedef double_linked_list[Condition].const_iterator const_iterator;

        # Functions:

        const_iterator cbegin()

        const_iterator cend()

        uint32 getNumConditions(Comparator comparator)

        void append(Condition condition)


cdef extern from "cpp/model/body.h" nogil:

    cdef cppclass IBody:
        pass


cdef extern from "cpp/model/body_empty.h" nogil:

    cdef cppclass EmptyBodyImpl"EmptyBody"(IBody):
        pass


cdef extern from "cpp/model/body_conjunctive.h" nogil:

    cdef cppclass ConjunctiveBodyImpl"ConjunctiveBody"(IBody):

        ctypedef const float32* threshold_const_iterator

        ctypedef const uint32* index_const_iterator

        uint32 getNumLeq()

        threshold_const_iterator leq_thresholds_cbegin()

        threshold_const_iterator leq_thresholds_cend()

        index_const_iterator leq_indices_cbegin()

        index_const_iterator leq_indices_cend()

        uint32 getNumGr()

        threshold_const_iterator gr_thresholds_cbegin()

        threshold_const_iterator gr_thresholds_cend()

        index_const_iterator gr_indices_cbegin()

        index_const_iterator gr_indices_cend()

        uint32 getNumEq()

        threshold_const_iterator eq_thresholds_cbegin()

        threshold_const_iterator eq_thresholds_cend()

        index_const_iterator eq_indices_cbegin()

        index_const_iterator eq_indices_cend()

        uint32 getNumNeq()

        threshold_const_iterator neq_thresholds_cbegin()

        threshold_const_iterator neq_thresholds_cend()

        index_const_iterator neq_indices_cbegin()

        index_const_iterator neq_indices_cend()


ctypedef void (*EmptyBodyVisitor)(const EmptyBodyImpl&)
ctypedef void (*ConjunctiveBodyVisitor)(const ConjunctiveBodyImpl&)


cdef extern from "cpp/model/head.h" nogil:

    cdef cppclass IHead:
        pass


cdef extern from "cpp/model/head_full.h" nogil:

    cdef cppclass FullHeadImpl"FullHead"(IHead):

        ctypedef const float64* score_const_iterator

        uint32 getNumElements()

        score_const_iterator scores_cbegin()

        score_const_iterator scores_cend()


cdef extern from "cpp/model/head_partial.h" nogil:

    cdef cppclass PartialHeadImpl"PartialHead"(IHead):

        ctypedef const float64* score_const_iterator

        ctypedef const uint32* index_const_iterator

        uint32 getNumElements()

        score_const_iterator scores_cbegin()

        score_const_iterator scores_cend()

        index_const_iterator indices_cbegin()

        index_const_iterator indices_cend()


ctypedef void (*FullHeadVisitor)(const FullHeadImpl&)
ctypedef void (*PartialHeadVisitor)(const PartialHeadImpl&)


cdef extern from "cpp/model/rule_model.h" nogil:

    cdef cppclass RuleModelImpl"RuleModel":

        void visit(EmptyBodyVisitor emptyBodyVisitor, ConjunctiveBodyVisitor conjunctiveBodyVisitor,
                   FullHeadVisitor fullHeadVisitor, PartialHeadVisitor partialHeadVisitor)


cdef extern from "cpp/model/model_builder.h" nogil:

    cdef cppclass IModelBuilder:

        # Functions:

        void setDefaultRule(const AbstractPrediction& prediction)

        void addRule(const ConditionList& conditions, const AbstractPrediction& prediction)

        unique_ptr[RuleModelImpl] build()


cdef class RuleModel:

    # Attributes:

    cdef unique_ptr[RuleModelImpl] model_ptr


cdef class ModelBuilder:

    # Attributes:

    cdef shared_ptr[IModelBuilder] model_builder_ptr

    # Functions:

    cdef RuleModel build(self)
