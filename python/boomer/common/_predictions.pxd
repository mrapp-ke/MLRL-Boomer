"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

classes that store the predictions of rules, as well as corresponding quality scores.
"""
from boomer.common._arrays cimport uint32, float64


cdef extern from "cpp/predictions.h" nogil:

    cdef cppclass Prediction:

        # Attributes:

        uint32* labelIndices_

        float64* predictedScores_

        # Functions:

        uint32 getNumElements()


    cdef cppclass PredictionCandidate(Prediction):
        pass
