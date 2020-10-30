from boomer.common._arrays cimport uint8, uint32, intp, float32, float64
from boomer.common._predictions cimport AbstractPrediction

from libcpp cimport bool
from libcpp.list cimport list as double_linked_list


cdef extern from "cpp/rules.h" nogil:

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


cdef class Body:

    # Functions:

    cdef bint covers(self, float32[::1] example)

    cdef bint covers_sparse(self, float32[::1] example_data, uint32[::1] example_indices, float32[::1] tmp_array1,
                            uint32[::1] tmp_array2, uint32 n)


cdef class EmptyBody(Body):

    # Functions:

    cdef bint covers(self, float32[::1] example)

    cdef bint covers_sparse(self, float32[::1] example_data, uint32[::1] example_indices, float32[::1] tmp_array1,
                            uint32[::1] tmp_array2, uint32 n)


cdef class ConjunctiveBody(Body):

    # Attributes:

    cdef readonly uint32[::1] leq_feature_indices

    cdef readonly float32[::1] leq_thresholds

    cdef readonly uint32[::1] gr_feature_indices

    cdef readonly float32[::1] gr_thresholds

    cdef readonly uint32[::1] eq_feature_indices

    cdef readonly float32[::1] eq_thresholds

    cdef readonly uint32[::1] neq_feature_indices

    cdef readonly float32[::1] neq_thresholds

    # Functions:

    cdef bint covers(self, float32[::1] example)

    cdef bint covers_sparse(self, float32[::1] example_data, uint32[::1] example_indices, float32[::1] tmp_array1,
                            uint32[::1] tmp_array2, uint32 n)


cdef class Head:

    # Functions:

    cdef void predict(self, float64[::1] predictions, uint8[::1] mask=*)


cdef class FullHead(Head):

    # Attributes:

    cdef readonly float64[::1] scores

    # Functions:

    cdef void predict(self, float64[::1] predictions, uint8[::1] mask=*)


cdef class PartialHead(Head):

    # Attributes:

    cdef readonly uint32[::1] label_indices

    cdef readonly float64[::1] scores

    # Functions:

    cdef void predict(self, float64[::1] predictions, uint8[::1] mask=*)


cdef class Rule:

    # Attributes:

    cdef readonly Body body

    cdef readonly Head head

    # Functions:

    cdef predict(self, float32[:, ::1] x, float64[:, ::1] predictions, uint8[:, ::1] mask=*)

    cdef predict_csr(self, float32[::1] x_data, uint32[::1] x_row_indices, uint32[::1] x_col_indices,
                     uint32 num_features, float32[::1] tmp_array1, uint32[::1] tmp_array2, uint32 n,
                     float64[:, ::1] predictions, uint8[:, ::1] mask=*)


cdef class RuleModel:

    # Functions:

    cdef void add_rule(self, Rule rule)

    cdef float64[:, ::1] predict(self, float32[:, ::1] x, uint32 num_labels)

    cdef float64[:, ::1] predict_csr(self, float32[::1] x_data, uint32[::1] x_row_indices, uint32[::1] x_col_indices,
                                     uint32 num_features, uint32 num_labels)


cdef class RuleList(RuleModel):

    # Attributes:

    cdef readonly bint use_mask

    cdef readonly list rules

    # Functions:

    cdef void add_rule(self, Rule rule)

    cdef float64[:, ::1] predict(self, float32[:, ::1] x, uint32 num_labels)

    cdef float64[:, ::1] predict_csr(self, float32[::1] x_data, uint32[::1] x_row_indices, uint32[::1] x_col_indices,
                                     uint32 num_features, uint32 num_labels)


cdef class ModelBuilder:

    # Functions:

    cdef void set_default_rule(self, AbstractPrediction* prediction)

    cdef void add_rule(self, AbstractPrediction* prediction, double_linked_list[Condition] conditions,
                       uint32[::1] num_conditions_per_comparator)

    cdef RuleModel build_model(self)


cdef class RuleListBuilder(ModelBuilder):

    # Attributes:

    cdef readonly bint use_mask

    cdef readonly bint default_rule_at_end

    cdef RuleList rule_list

    cdef Rule default_rule

    # Functions:

    cdef void set_default_rule(self, AbstractPrediction* prediction)

    cdef void add_rule(self, AbstractPrediction* prediction, double_linked_list[Condition] conditions,
                       uint32[::1] num_conditions_per_comparator)

    cdef RuleModel build_model(self)
