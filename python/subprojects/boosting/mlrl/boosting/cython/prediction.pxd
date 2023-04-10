from libcpp cimport bool


cdef extern from "boosting/prediction/predictor_binary_example_wise.hpp" namespace "boosting" nogil:

    cdef cppclass IExampleWiseBinaryPredictorConfig:

        # Functions:

        bool isBasedOnProbabilities() const

        IExampleWiseBinaryPredictorConfig& setBasedOnProbabilities(bool basedOnProbabilities)


cdef extern from "boosting/prediction/predictor_binary_gfm.hpp" namespace "boosting" nogil:

    cdef cppclass IGfmBinaryPredictorConfig:

        # Functions:

        bool isBasedOnProbabilities() const

        IGfmBinaryPredictorConfig& setBasedOnProbabilities(bool basedOnProbabilities)


cdef extern from "boosting/prediction/predictor_binary_label_wise.hpp" namespace "boosting" nogil:

    cdef cppclass ILabelWiseBinaryPredictorConfig:

        # Functions:

        bool isBasedOnProbabilities() const

        ILabelWiseBinaryPredictorConfig& setBasedOnProbabilities(bool basedOnProbabilities)


cdef class ExampleWiseBinaryPredictorConfig:

    # Attributes:

    cdef IExampleWiseBinaryPredictorConfig* config_ptr


cdef class GfmBinaryPredictorConfig:

    # Attributes:

    cdef IGfmBinaryPredictorConfig* config_ptr


cdef class LabelWiseBinaryPredictorConfig:

    # Attributes:

    cdef ILabelWiseBinaryPredictorConfig* config_ptr
