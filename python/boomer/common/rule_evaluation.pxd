"""
@author Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides classes that store the predictions of rules, as well as corresponding quality scores.
"""
from boomer.common._arrays cimport intp, float64


cdef extern from "cpp/rule_evaluation.h" namespace "rule_evaluation":

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
