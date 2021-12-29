from mlrl.common.cython._types cimport uint8, uint32, float64
from mlrl.common.cython._measures cimport ISimilarityMeasureFactory
from mlrl.common.cython.measures cimport SimilarityMeasureFactory
from mlrl.common.cython.output cimport AbstractBinaryPredictor, AbstractNumericalPredictor, \
    ProbabilityPredictorFactory, IProbabilityPredictorFactory, RegressionPredictorFactory, \
    IRegressionPredictorFactory, ClassificationPredictorFactory, IClassificationPredictorFactory

from libcpp.memory cimport unique_ptr


cdef extern from "boosting/output/predictor_probability_label_wise.hpp" namespace "boosting" nogil:

    cdef cppclass IProbabilityFunctionFactory:
        pass


    cdef cppclass LogisticFunctionFactoryImpl"boosting::LogisticFunctionFactory"(IProbabilityFunctionFactory):
        pass


    cdef cppclass LabelWiseProbabilityPredictorFactoryImpl"boosting::LabelWiseProbabilityPredictorFactory"(
            IProbabilityPredictorFactory):

        # Constructors:

        LabelWiseProbabilityPredictorFactoryImpl(unique_ptr[IProbabilityFunctionFactory] probabilityFunctionFactoryPtr,
                                                 uint32 numThreads) except +


cdef extern from "boosting/output/predictor_regression_label_wise.hpp" namespace "boosting" nogil:

    cdef cppclass LabelWiseRegressionPredictorFactoryImpl"boosting::LabelWiseRegressionPredictorFactory"(
            IRegressionPredictorFactory):

        # Constructors:

        LabelWiseRegressionPredictorFactoryImpl(uint32 numThreads) except +


cdef extern from "boosting/output/predictor_classification_label_wise.hpp" namespace "boosting" nogil:

    cdef cppclass LabelWiseClassificationPredictorFactoryImpl"boosting::LabelWiseClassificationPredictorFactory"(
            IClassificationPredictorFactory):

        # Constructors:

        LabelWiseClassificationPredictorFactoryImpl(float64 threshold, uint32 numThreads) except +


cdef extern from "boosting/output/predictor_classification_example_wise.hpp" namespace "boosting" nogil:

    cdef cppclass ExampleWiseClassificationPredictorFactoryImpl"boosting::ExampleWiseClassificationPredictorFactory"(
            IClassificationPredictorFactory):

        # Constructors:

        ExampleWiseClassificationPredictorFactoryImpl(unique_ptr[ISimilarityMeasureFactory] similarityMeasureFactoryPtr,
                                                      uint32 numThreads) except +


cdef class ProbabilityFunctionFactory:

    # Attributes:

    cdef unique_ptr[IProbabilityFunctionFactory] probability_function_factory_ptr


cdef class LogisticFunctionFactory(ProbabilityFunctionFactory):
    pass


cdef class LabelWiseProbabilityPredictorFactory(ProbabilityPredictorFactory):

    # Attributes:

    cdef ProbabilityFunctionFactory probability_function_factory

    cdef uint32 num_threads


cdef class LabelWiseRegressionPredictorFactory(RegressionPredictorFactory):

    # Attributes

    cdef uint32 num_threads


cdef class LabelWiseClassificationPredictorFactory(ClassificationPredictorFactory):

    # Attributes:

    cdef float64 threshold

    cdef uint32 num_threads


cdef class ExampleWiseClassificationPredictorFactory(ClassificationPredictorFactory):

    # Attributes

    cdef SimilarityMeasureFactory similarity_measure_factory

    cdef uint32 num_threads
