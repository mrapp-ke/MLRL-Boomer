"""
@author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from libcpp.cast cimport dynamic_cast

from cython.operator cimport dereference

from mlrl.common.cython._serialization cimport check_pickled_state, image_to_bytes

from abc import ABC, abstractmethod


cdef class MarginalProbabilityCalibrationModel:
    """
    A model that may be used for the calibration of marginal probabilities.
    """

    cdef IMarginalProbabilityCalibrationModel* get_marginal_probability_calibration_model_ptr(self):
        pass


cdef class JointProbabilityCalibrationModel:
    """
    A model that may be used for the calibration of joint probabilities.
    """

    cdef IJointProbabilityCalibrationModel* get_joint_probability_calibration_model_ptr(self):
        pass


class NoProbabilityCalibrationModel(ABC):
    """
    Defines an interface for all models for the calibration of probabilities that do not make any adjustments.
    """


cdef class NoMarginalProbabilityCalibrationModel(MarginalProbabilityCalibrationModel, NoProbabilityCalibrationModel):
    """
    A model for the calibration of marginal probabilities that does not make any adjustments.
    """

    cdef IMarginalProbabilityCalibrationModel* get_marginal_probability_calibration_model_ptr(self):
        return self.probability_calibration_model_ptr.get()

    def __reduce__(self):
        return (NoMarginalProbabilityCalibrationModel, (), ())

    def __setstate__(self, state):
        self.probability_calibration_model_ptr = createNoProbabilityCalibrationModel()


cdef class NoJointProbabilityCalibrationModel(JointProbabilityCalibrationModel, NoProbabilityCalibrationModel):
    """
    A model for the calibration of joint probabilities that does not make any adjustments.
    """

    cdef IJointProbabilityCalibrationModel* get_joint_probability_calibration_model_ptr(self):
        return self.probability_calibration_model_ptr.get()

    def __reduce__(self):
        return (NoJointProbabilityCalibrationModel, (), ())

    def __setstate__(self, state):
        self.probability_calibration_model_ptr = createNoProbabilityCalibrationModel()


class IsotonicProbabilityCalibrationModelVisitor(ABC):
    """
    Defines an interface for all visitors that allow to handle the bins stores by a model for the calibration of
    probabilities via isotonic regression.
    """

    @abstractmethod
    def visit_bin(self, list_index: int, threshold: float, probability: float):
        """
        Must be implemented by subclasses in order to handle a bin.

        :param list_index:  The index of the list, the bin corresponds to
        :param threshold:   The threshold of the bin
        :param probability: The probability of the bin
        """


class IsotonicProbabilityCalibrationModel(ABC):
    """
    Defines an interface for all models for the calibration of probabilities via isotonic regression.
    """

    @abstractmethod
    def visit(self, visitor: IsotonicProbabilityCalibrationModelVisitor):
        """
        Invokes a given visitor for each bin that is stored by the calibration model.

        :param visitor: The visitor to be invoked
        """


cdef class IsotonicMarginalProbabilityCalibrationModel(MarginalProbabilityCalibrationModel,
                                                       IsotonicProbabilityCalibrationModel):
    """
    A model for the calibration of marginal probabilities via isotonic regression.
    """

    def visit(self, visitor: IsotonicProbabilityCalibrationModelVisitor):
        self.visitor = visitor
        self.probability_calibration_model_ptr.get().visit(
            wrapBinVisitor(<void*>self, <BinCythonVisitor>self.__visit_bin))
        self.visitor = None

    cdef IMarginalProbabilityCalibrationModel* get_marginal_probability_calibration_model_ptr(self):
        return self.probability_calibration_model_ptr.get()

    cdef __visit_bin(self, uint32 list_index, float64 threshold, float64 probability):
        self.visitor.visit_bin(list_index, threshold, probability)

    def __reduce__(self):
        cdef bytes state = image_to_bytes(
            serializeMarginalProbabilityCalibrationModel(dereference(self.probability_calibration_model_ptr.get())))
        return (IsotonicMarginalProbabilityCalibrationModel, (), state)

    def __setstate__(self, state):
        check_pickled_state(state, 'IsotonicMarginalProbabilityCalibrationModel')
        cdef const uint8[::1] data = state
        cdef const uint8* data_ptr = &data[0] if data.shape[0] > 0 else NULL
        cdef unique_ptr[IMarginalProbabilityCalibrationModel] model_ptr = (
            deserializeMarginalProbabilityCalibrationModel(data_ptr, data.shape[0]))
        cdef IIsotonicProbabilityCalibrationModel* isotonic_model_ptr = (
            dynamic_cast[IsotonicProbabilityCalibrationModelPtr](model_ptr.get()))

        if isotonic_model_ptr == NULL:
            raise ValueError('The pickled data does not contain an object of type '
                             + 'IsotonicMarginalProbabilityCalibrationModel')

        model_ptr.release()
        self.probability_calibration_model_ptr = unique_ptr[IIsotonicProbabilityCalibrationModel](isotonic_model_ptr)


cdef class IsotonicJointProbabilityCalibrationModel(JointProbabilityCalibrationModel,
                                                    IsotonicProbabilityCalibrationModel):
    """
    A model for the calibration of joint probabilities via isotonic regression.
    """

    def visit(self, visitor: IsotonicProbabilityCalibrationModelVisitor):
        self.visitor = visitor
        self.probability_calibration_model_ptr.get().visit(
            wrapBinVisitor(<void*>self, <BinCythonVisitor>self.__visit_bin))
        self.visitor = None

    cdef IJointProbabilityCalibrationModel* get_joint_probability_calibration_model_ptr(self):
        return self.probability_calibration_model_ptr.get()

    cdef __visit_bin(self, uint32 list_index, float64 threshold, float64 probability):
        self.visitor.visit_bin(list_index, threshold, probability)

    def __reduce__(self):
        cdef bytes state = image_to_bytes(
            serializeJointProbabilityCalibrationModel(dereference(self.probability_calibration_model_ptr.get())))
        return (IsotonicJointProbabilityCalibrationModel, (), state)

    def __setstate__(self, state):
        check_pickled_state(state, 'IsotonicJointProbabilityCalibrationModel')
        cdef const uint8[::1] data = state
        cdef const uint8* data_ptr = &data[0] if data.shape[0] > 0 else NULL
        cdef unique_ptr[IJointProbabilityCalibrationModel] model_ptr = (
            deserializeJointProbabilityCalibrationModel(data_ptr, data.shape[0]))
        cdef IIsotonicProbabilityCalibrationModel* isotonic_model_ptr = (
            dynamic_cast[IsotonicProbabilityCalibrationModelPtr](model_ptr.get()))

        if isotonic_model_ptr == NULL:
            raise ValueError('The pickled data does not contain an object of type '
                             + 'IsotonicJointProbabilityCalibrationModel')

        model_ptr.release()
        self.probability_calibration_model_ptr = unique_ptr[IIsotonicProbabilityCalibrationModel](isotonic_model_ptr)
