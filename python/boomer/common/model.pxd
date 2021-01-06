from boomer.common._types cimport uint32, intp, float32
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


cdef extern from "cpp/model/rule_model.h" nogil:

    cdef cppclass RuleModelImpl"RuleModel":
        pass


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
