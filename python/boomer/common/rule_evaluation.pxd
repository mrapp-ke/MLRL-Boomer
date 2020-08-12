from boomer.common._arrays cimport intp, float64
from boomer.common.input_data cimport AbstractRandomAccessLabelMatrix

from libcpp.memory cimport shared_ptr


cdef extern from "cpp/rule_evaluation.h" nogil:

    cdef cppclass DefaultPrediction:

        # Constructors:

        DefaultPrediction(intp numPredictions, float64* predictedScores) except +

        # Attributes:

        intp numPredictions_

        float64* predictedScores_


    cdef cppclass Prediction(DefaultPrediction):

        # Constructors:

        Prediction(intp numPredictions, float64* predictedScores, float64 overallQualityScore) except +

        # Attributes:

        float64 overallQualityScore_


    cdef cppclass LabelWisePrediction(Prediction):

        # Constructors:

        LabelWisePrediction(intp numPredictions, float64* predictedScores, float64* qualityScores,
                            float64 overallQualityScore) except +

        # Attributes:

        float64* qualityScores_


    cdef cppclass AbstractDefaultRuleEvaluation:

        # Functions:

        DefaultPrediction* calculateDefaultPrediction(AbstractRandomAccessLabelMatrix* labelMatrix) except +


cdef class DefaultRuleEvaluation:

    # Attributes:

    cdef shared_ptr[AbstractDefaultRuleEvaluation] default_rule_evaluation_ptr
