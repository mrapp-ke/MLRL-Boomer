"""
@author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from mlrl.common.cython._types cimport uint32

from libcpp.utility cimport move
from libcpp.memory cimport unique_ptr, make_unique


cdef class ProbabilityFunctionFactory:
    """
    A wrapper for the pure virtual C++ class `IProbabilityFunctionFactory`.
    """
    pass


cdef class LogisticFunctionFactory(ProbabilityFunctionFactory):
    """
    A wrapper for the C++ class `LogisticFunctionFactory`.
    """

    def __cinit__(self):
        self.probability_function_factory_ptr = <unique_ptr[IProbabilityFunctionFactory]>make_unique[LogisticFunctionFactoryImpl]()

    def __reduce__(self):
        return (LogisticFunctionFactory, ())


cdef class LabelWiseProbabilityPredictorFactory(ProbabilityPredictorFactory):

    def __cinit__(self, uint32 num_labels, ProbabilityFunctionFactory probability_function_factory not None,
                  uint32 num_threads):
        self.num_labels = num_labels
        self.probability_function_factory = probability_function_factory
        self.num_threads = num_threads
        self.predictor_factory_ptr = <unique_ptr[IProbabilityPredictorFactory]>make_unique[LabelWiseProbabilityPredictorFactoryImpl](
            move(probability_function_factory.probability_function_factory_ptr), num_threads)

    def __reduce__(self):
        return (LabelWiseProbabilityPredictorFactory, (self.num_labels, self.probability_function_factory,
                                                       self.num_threads))


cdef class LabelWiseRegressionPredictorFactory(RegressionPredictorFactory):
    """
    A wrapper for the C++ class `LabelWiseRegressionPredictorFactory`.
    """

    def __cinit__(self, uint32 num_labels, uint32 num_threads):
        """
        :param num_labels:  The total number of available labels
        :param num_threads: The number of CPU threads to be used to make predictions for different query examples in
                            parallel. Must be at least 1
        """
        self.num_labels = num_labels
        self.num_threads = num_threads
        self.predictor_factory_ptr = <unique_ptr[IRegressionPredictorFactory]>make_unique[LabelWiseRegressionPredictorFactoryImpl](
            num_threads)

    def __reduce__(self):
        return (LabelWiseRegressionPredictorFactory, (self.num_labels, self.num_threads))


cdef class LabelWiseClassificationPredictorFactory(ClassificationPredictorFactory):
    """
    A wrapper for the C++ class `LabelWiseClassificationPredictorFactory`.
    """

    def __cinit__(self, uint32 num_labels, float64 threshold, uint32 num_threads):
        """
        :param num_labels:  The total number of available labels
        :param thresholds:  The threshold to be used for making predictions
        :param num_threads: The number of CPU threads to be used to make predictions for different query examples in
                            parallel. Must be at least 1
        """
        self.num_labels = num_labels
        self.threshold = threshold
        self.num_threads = num_threads
        self.predictor_factory_ptr = <unique_ptr[IClassificationPredictorFactory]>make_unique[LabelWiseClassificationPredictorFactoryImpl](
            threshold, num_threads)

    def __reduce__(self):
        return (LabelWiseClassificationPredictorFactory, (self.num_labels, self.threshold, self.num_threads))


cdef class ExampleWiseClassificationPredictorFactory(ClassificationPredictorFactory):
    """
    A wrapper for the C++ class `ExampleWiseClassificationPredictorFactory`.
    """

    def __cinit__(self, uint32 num_labels, SimilarityMeasureFactory similarity_measure_factory not None,
                  uint32 num_threads):
        """
        :param num_labels:                  The total number of available labels
        :param similarity_measure_factory:  The `SimilarityMeasureFactory` to be used
        :param num_threads:                 The number of CPU threads to be used to make predictions for different query
                                            examples in parallel. Must be at least 1
        """
        self.num_labels = num_labels
        self.similarity_measure_factory = similarity_measure_factory
        self.num_threads = num_threads
        self.predictor_factory_ptr = <unique_ptr[IClassificationPredictorFactory]>make_unique[ExampleWiseClassificationPredictorFactoryImpl](
            move(similarity_measure_factory.get_similarity_measure_factory_ptr()), num_threads)

    def __reduce__(self):
        return (ExampleWiseClassificationPredictorFactory, (self.num_labels, self.measure, self.num_threads))
