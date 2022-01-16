from mlrl.common.cython._types cimport uint32


cdef extern from "boosting/output/predictor_classification_example_wise.hpp" namespace "boosting" nogil:

    cdef cppclass ExampleWiseClassificationPredictorConfigImpl"boosting::ExampleWiseClassificationPredictorConfig":

        # Functions:

        uint32 getNumThreads() const

        ExampleWiseClassificationPredictorConfigImpl& setNumThreads(uint32 numThreads) except +


cdef extern from "boosting/output/predictor_classification_label_wise.hpp" namespace "boosting" nogil:

    cdef cppclass LabelWiseClassificationPredictorConfigImpl"boosting::LabelWiseClassificationPredictorConfig":

        # Functions:

        uint32 getNumThreads() const

        LabelWiseClassificationPredictorConfigImpl& setNumThreads(uint32 numThreads) except +


cdef extern from "boosting/output/predictor_regression_label_wise.hpp" namespace "boosting" nogil:

    cdef cppclass LabelWiseRegressionPredictorConfigImpl"boosting::LabelWiseRegressionPredictorConfig":

        # Functions:

        uint32 getNumThreads() const

        LabelWiseRegressionPredictorConfigImpl& setNumThreads(uint32 numThreads) except +


cdef extern from "boosting/output/predictor_probability_label_wise.hpp" namespace "boosting" nogil:

    cdef cppclass LabelWiseProbabilityPredictorConfigImpl"boosting::LabelWiseProbabilityPredictorConfig":

        # Functions:

        uint32 getNumThreads() const

        LabelWiseProbabilityPredictorConfigImpl& setNumThreads(uint32 numThreads) except +


cdef class ExampleWiseClassificationPredictorConfig:

    # Attributes:

    cdef ExampleWiseClassificationPredictorConfigImpl* config_ptr


cdef class LabelWiseClassificationPredictorConfig:

    # Attributes:

    cdef LabelWiseClassificationPredictorConfigImpl* config_ptr


cdef class LabelWiseRegressionPredictorConfig:

    # Attributes:

    cdef LabelWiseRegressionPredictorConfigImpl* config_ptr


cdef class LabelWiseProbabilityPredictorConfig:

    # Attributes:

    cdef LabelWiseProbabilityPredictorConfigImpl* config_ptr
