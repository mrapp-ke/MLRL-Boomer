from mlrl.common.cython._measures cimport IEvaluationMeasureFactory, ISimilarityMeasureFactory
from mlrl.common.cython.measures cimport EvaluationMeasureFactory

from libcpp.memory cimport unique_ptr


cdef extern from "boosting/losses/loss_label_wise.hpp" namespace "boosting" nogil:

    cdef cppclass ILabelWiseLossFactory(IEvaluationMeasureFactory, ISimilarityMeasureFactory):
        pass


cdef extern from "boosting/losses/loss_label_wise_logistic.hpp" namespace "boosting" nogil:

    cdef cppclass LabelWiseLogisticLossFactoryImpl"boosting::LabelWiseLogisticLossFactory"(ILabelWiseLossFactory):
        pass


cdef extern from "boosting/losses/loss_label_wise_squared_error.hpp" namespace "boosting" nogil:

    cdef cppclass LabelWiseSquaredErrorLossFactoryImpl"boosting::LabelWiseSquaredErrorLossFactory"(
            ILabelWiseLossFactory):
        pass


cdef extern from "boosting/losses/loss_label_wise_squared_hinge.hpp" namespace "boosting" nogil:

    cdef cppclass LabelWiseSquaredHingeLossFactoryImpl"boosting::LabelWiseSquaredHingeLossFactory"(
            ILabelWiseLossFactory):
        pass


cdef class LabelWiseLossFactory(EvaluationMeasureFactory):

    # Attributes:

    cdef unique_ptr[ILabelWiseLossFactory] loss_factory_ptr


cdef class LabelWiseLogisticLossFactory(LabelWiseLossFactory):
    pass


cdef class LabelWiseSquaredErrorLossFactory(LabelWiseLossFactory):
    pass


cdef class LabelWiseSquaredHingeLossFactory(LabelWiseLossFactory):
    pass
