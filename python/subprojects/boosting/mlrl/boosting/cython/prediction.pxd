from libcpp cimport bool


cdef extern from "mlrl/boosting/prediction/predictor_probability_output_wise.hpp" namespace "boosting" nogil:

    cdef cppclass IOutputWiseProbabilityPredictorConfig:

        # Functions:

        bool isProbabilityCalibrationModelUsed() const

        IOutputWiseProbabilityPredictorConfig& setUseProbabilityCalibrationModel(bool useProbabilityCalibrationModel)


cdef extern from "mlrl/boosting/prediction/predictor_probability_marginalized.hpp" namespace "boosting" nogil:

    cdef cppclass IMarginalizedProbabilityPredictorConfig:

        # Functions:

        bool isProbabilityCalibrationModelUsed() const

        IMarginalizedProbabilityPredictorConfig& setUseProbabilityCalibrationModel(bool useProbabilityCalibrationModel)


cdef extern from "mlrl/boosting/prediction/predictor_binary_example_wise.hpp" namespace "boosting" nogil:

    cdef cppclass IExampleWiseBinaryPredictorConfig:

        # Functions:

        bool isBasedOnProbabilities() const

        IExampleWiseBinaryPredictorConfig& setBasedOnProbabilities(bool basedOnProbabilities)

        bool isProbabilityCalibrationModelUsed() const

        IExampleWiseBinaryPredictorConfig& setUseProbabilityCalibrationModel(bool useProbabilityCalibrationModel)


cdef extern from "mlrl/boosting/prediction/predictor_binary_output_wise.hpp" namespace "boosting" nogil:

    cdef cppclass IOutputWiseBinaryPredictorConfig:

        # Functions:

        bool isBasedOnProbabilities() const

        IOutputWiseBinaryPredictorConfig& setBasedOnProbabilities(bool basedOnProbabilities)

        bool isProbabilityCalibrationModelUsed() const

        IExampleWiseBinaryPredictorConfig& setUseProbabilityCalibrationModel(bool useProbabilityCalibrationModel)


cdef extern from "mlrl/boosting/prediction/predictor_binary_gfm.hpp" namespace "boosting" nogil:

    cdef cppclass IGfmBinaryPredictorConfig:

        # Functions:

        bool isProbabilityCalibrationModelUsed() const

        IGfmBinaryPredictorConfig& setUseProbabilityCalibrationModel(bool useProbabilityCalibrationModel)


cdef class OutputWiseProbabilityPredictorConfig:

    # Attributes:

    cdef IOutputWiseProbabilityPredictorConfig* config_ptr


cdef class MarginalizedProbabilityPredictorConfig:

    # Attributes:

    cdef IMarginalizedProbabilityPredictorConfig* config_ptr


cdef class ExampleWiseBinaryPredictorConfig:

    # Attributes:

    cdef IExampleWiseBinaryPredictorConfig* config_ptr


cdef class OutputWiseBinaryPredictorConfig:

    # Attributes:

    cdef IOutputWiseBinaryPredictorConfig* config_ptr


cdef class GfmBinaryPredictorConfig:

    # Attributes:

    cdef IGfmBinaryPredictorConfig* config_ptr
