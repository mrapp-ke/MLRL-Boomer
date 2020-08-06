from boomer.common._arrays cimport intp, float64
from boomer.common.input_data cimport LabelMatrix, AbstractLabelMatrix


cdef extern from "cpp/rule_evaluation.h":

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

        DefaultPrediction* calculateDefaultPrediction(AbstractLabelMatrix* labelMatrix) nogil except +


cdef class DefaultRuleEvaluation:

    # Attributes:

    cdef AbstractDefaultRuleEvaluation* default_rule_evaluation

    # Functions:

    cdef DefaultPrediction* calculate_default_prediction(self, LabelMatrix label_matrix) nogil
