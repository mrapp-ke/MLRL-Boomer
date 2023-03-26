"""
@author: Michael Rapp (michael.rapp.ml@gmail.com)
"""


cdef class ProbabilityCalibrationModel:
    """
    A model that may be used for the calibration of probabilities.
    """

    cdef IProbabilityCalibrationModel* get_probability_calibration_model_ptr(self):
        pass


cdef class NoProbabilityCalibrationModel(ProbabilityCalibrationModel):
    """
    Does not provide any information about the label space.
    """

    cdef IProbabilityCalibrationModel* get_probability_calibration_model_ptr(self):
        return self.probability_calibration_model_ptr.get()

    def __reduce__(self):
        return (NoProbabilityCalibrationModel, (), ())

    def __setstate__(self, state):
        self.probability_calibration_model_ptr = createNoProbabilityCalibrationModel()
