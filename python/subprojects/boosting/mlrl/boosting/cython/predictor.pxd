from mlrl.common.cython._types cimport uint32


cdef extern from "boosting/output/predictor_classification_example_wise.hpp" namespace "boosting" nogil:

    cdef cppclass IExampleWiseClassificationPredictorConfig:

        # Functions:

        uint32 getNumThreads() const

        IExampleWiseClassificationPredictorConfig& setNumThreads(uint32 numThreads) except +


cdef extern from "boosting/output/predictor_classification_label_wise.hpp" namespace "boosting" nogil:

    cdef cppclass ILabelWiseClassificationPredictorConfig:

        # Functions:

        uint32 getNumThreads() const

        ILabelWiseClassificationPredictorConfig& setNumThreads(uint32 numThreads) except +


cdef extern from "boosting/output/predictor_regression_label_wise.hpp" namespace "boosting" nogil:

    cdef cppclass ILabelWiseRegressionPredictorConfig:

        # Functions:

        uint32 getNumThreads() const

        ILabelWiseRegressionPredictorConfig& setNumThreads(uint32 numThreads) except +


cdef extern from "boosting/output/predictor_probability_label_wise.hpp" namespace "boosting" nogil:

    cdef cppclass ILabelWiseProbabilityPredictorConfig:

        # Functions:

        uint32 getNumThreads() const

        ILabelWiseProbabilityPredictorConfig& setNumThreads(uint32 numThreads) except +


cdef class ExampleWiseClassificationPredictorConfig:

    # Attributes:

    cdef IExampleWiseClassificationPredictorConfig* config_ptr


cdef class LabelWiseClassificationPredictorConfig:

    # Attributes:

    cdef ILabelWiseClassificationPredictorConfig* config_ptr


cdef class LabelWiseRegressionPredictorConfig:

    # Attributes:

    cdef ILabelWiseRegressionPredictorConfig* config_ptr


cdef class LabelWiseProbabilityPredictorConfig:

    # Attributes:

    cdef ILabelWiseProbabilityPredictorConfig* config_ptr
