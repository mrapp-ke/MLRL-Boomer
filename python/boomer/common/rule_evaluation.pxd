from boomer.common._predictions cimport Prediction
from boomer.common.input_data cimport AbstractLabelMatrix

from libcpp.memory cimport shared_ptr


cdef extern from "cpp/rule_evaluation.h" nogil:

    cdef cppclass AbstractDefaultRuleEvaluation:

        # Functions:

        Prediction* calculateDefaultPrediction(AbstractLabelMatrix* labelMatrix) except +


cdef class DefaultRuleEvaluation:

    # Attributes:

    cdef shared_ptr[AbstractDefaultRuleEvaluation] default_rule_evaluation_ptr
