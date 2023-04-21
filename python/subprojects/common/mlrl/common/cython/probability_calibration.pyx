"""
@author: Michael Rapp (michael.rapp.ml@gmail.com)
"""


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


cdef class NoMarginalProbabilityCalibrationModel(MarginalProbabilityCalibrationModel):
    """
    A model for the calibration of marginal probabilities that does not make any adjustments.
    """

    cdef IMarginalProbabilityCalibrationModel* get_marginal_probability_calibration_model_ptr(self):
        return self.probability_calibration_model_ptr.get()

    def __reduce__(self):
        return (NoMarginalProbabilityCalibrationModel, (), ())

    def __setstate__(self, state):
        self.probability_calibration_model_ptr = createNoMarginalProbabilityCalibrationModel()


cdef class NoJointProbabilityCalibrationModel(JointProbabilityCalibrationModel):
    """
    A model for the calibration of joint probabilities that does not make any adjustments.
    """

    cdef IJointProbabilityCalibrationModel* get_joint_probability_calibration_model_ptr(self):
        return self.probability_calibration_model_ptr.get()

    def __reduce__(self):
        return (NoJointProbabilityCalibrationModel, (), ())

    def __setstate__(self, state):
        self.probability_calibration_model_ptr = createNoJointProbabilityCalibrationModel()


cdef class IsotonicMarginalProbabilityCalibrationModel(MarginalProbabilityCalibrationModel):
    """
    A model for the calibration of marginal probabilities via isotonic regression.
    """

    cdef IMarginalProbabilityCalibrationModel* get_marginal_probability_calibration_model_ptr(self):
        return self.probability_calibration_model_ptr.get()
    
    def __reduce(self):
        return (IsotonicMarginalProbabilityCalibrationModel, (), ())

    def __setstate(self, state):
        self.probability_calibration_model_ptr = createIsotonicMarginalProbabilityCalibrationModel()


cdef class IsotonicJointProbabilityCalibrationModel(JointProbabilityCalibrationModel):
    """
    A model for the calibration of joint probabilities via isotonic regression.
    """

    cdef IJointProbabilityCalibrationModel* get_joint_probability_calibration_model_ptr(self):
        return self.probability_calibration_model_ptr.get()
    
    def __reduce(self):
        return (IsotonicJointProbabilityCalibrationModel, (), ())

    def __setstate(self, state):
        self.probability_calibration_model_ptr = createIsotonicJointProbabilityCalibrationModel()
