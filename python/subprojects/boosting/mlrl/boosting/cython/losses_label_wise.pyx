"""
@author Michael Rapp (michael.rapp.ml@gmail.com)
"""
from libcpp.utility cimport move
from libcpp.memory cimport make_unique


cdef class LabelWiseLossFactory(EvaluationMeasureFactory):
    """
    A wrapper for the pure virtual C++ class `ILabelWiseLossFactory`.
    """

    cdef unique_ptr[IEvaluationMeasureFactory] get_evaluation_measure_factory_ptr(self):
        return <unique_ptr[IEvaluationMeasureFactory]>move(self.loss_factory_ptr)

    cdef unique_ptr[ISimilarityMeasureFactory] get_similarity_measure_factory_ptr(self):
        return <unique_ptr[ISimilarityMeasureFactory]>move(self.loss_factory_ptr)


cdef class LabelWiseLogisticLossFactory(LabelWiseLossFactory):
    """
    A wrapper for the C++ class `LabelWiseLogisticLossFactory`.
    """

    def __cinit__(self):
        self.loss_factory_ptr = <unique_ptr[ILabelWiseLossFactory]>make_unique[LabelWiseLogisticLossFactoryImpl]()

    def __reduce__(self):
        return (LabelWiseLogisticLossFactory, ())


cdef class LabelWiseSquaredErrorLossFactory(LabelWiseLossFactory):
    """
    A wrapper for the C++ class `LabelWiseSquaredErrorLossFactory`.
    """

    def __cinit__(self):
        self.loss_factory_ptr = <unique_ptr[ILabelWiseLossFactory]>make_unique[LabelWiseSquaredErrorLossFactoryImpl]()

    def __reduce__(self):
        return (LabelWiseSquaredErrorLossFactory, ())


cdef class LabelWiseSquaredHingeLossFactory(LabelWiseLossFactory):
    """
    A wrapper for the C++ class `LabelWiseSquaredHingeLossFactory`.
    """

    def __cinit__(self):
        self.loss_factory_ptr = <unique_ptr[ILabelWiseLossFactory]>make_unique[LabelWiseSquaredHingeLossFactoryImpl]()

    def __reduce__(self):
        return (LabelWiseSquaredHingeLossFactory, ())
