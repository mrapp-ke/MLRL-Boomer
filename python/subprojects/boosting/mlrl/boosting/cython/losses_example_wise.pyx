"""
@author Michael Rapp (michael.rapp.ml@gmail.com)
"""
from libcpp.utility cimport move
from libcpp.memory cimport make_unique


cdef class ExampleWiseLossFactory(EvaluationMeasureFactory):
    """
    A wrapper for the pure virtual C++ class `IExampleWiseLossFactory`.
    """

    cdef unique_ptr[IEvaluationMeasureFactory] get_evaluation_measure_factory_ptr(self):
        return <unique_ptr[IEvaluationMeasureFactory]>move(self.loss_factory_ptr)

    cdef unique_ptr[ISimilarityMeasureFactory] get_similarity_measure_factory_ptr(self):
        return <unique_ptr[ISimilarityMeasureFactory]>move(self.loss_factory_ptr)


cdef class ExampleWiseLogisticLossFactory(ExampleWiseLossFactory):
    """
    A wrapper for the C++ class `ExampleWiseLogisticLossFactory`.
    """

    def __cinit__(self):
        self.loss_factory_ptr = <unique_ptr[IExampleWiseLossFactory]>make_unique[ExampleWiseLogisticLossFactoryImpl]()

    def __reduce__(self):
        return (ExampleWiseLogisticLossFactory, ())
