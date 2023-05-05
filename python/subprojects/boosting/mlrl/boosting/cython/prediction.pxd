from libcpp cimport bool


cdef extern from "boosting/prediction/predictor_binary_example_wise.hpp" namespace "boosting" nogil:

    cdef cppclass IExampleWiseBinaryPredictorConfig:

        # Functions:

        bool isBasedOnProbabilities() const

        IExampleWiseBinaryPredictorConfig& setBasedOnProbabilities(bool basedOnProbabilities)

        bool isProbabilityCalibrationModelUsed() const

        IExampleWiseBinaryPredictorConfig& setUseProbabilityCalibrationModel(bool useProbabilityCalibrationModel)


cdef extern from "boosting/prediction/predictor_binary_label_wise.hpp" namespace "boosting" nogil:

    cdef cppclass ILabelWiseBinaryPredictorConfig:

        # Functions:

        bool isBasedOnProbabilities() const

        ILabelWiseBinaryPredictorConfig& setBasedOnProbabilities(bool basedOnProbabilities)

        bool isProbabilityCalibrationModelUsed() const

        IExampleWiseBinaryPredictorConfig& setUseProbabilityCalibrationModel(bool useProbabilityCalibrationModel)


cdef class ExampleWiseBinaryPredictorConfig:

    # Attributes:

    cdef IExampleWiseBinaryPredictorConfig* config_ptr


cdef class LabelWiseBinaryPredictorConfig:

    # Attributes:

    cdef ILabelWiseBinaryPredictorConfig* config_ptr
