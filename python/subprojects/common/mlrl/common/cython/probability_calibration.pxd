from libcpp.cast cimport dynamic_cast
from libcpp.memory cimport unique_ptr


cdef extern from "common/prediction/probability_calibration.hpp" nogil:

    cdef cppclass IMarginalProbabilityCalibrationModel:
        pass


cdef extern from "common/prediction/probability_calibration_no.hpp" nogil:

    cdef cppclass INoProbabilityCalibrationModel(IMarginalProbabilityCalibrationModel):
        pass


    unique_ptr[INoProbabilityCalibrationModel] createNoProbabilityCalibrationModel()


ctypedef INoProbabilityCalibrationModel* NoProbabilityCalibrationModelPtr


cdef class MarginalProbabilityCalibrationModel:

    # Functions:

    cdef IMarginalProbabilityCalibrationModel* get_marginal_probability_calibration_model_ptr(self)


cdef class NoProbabilityCalibrationModel(MarginalProbabilityCalibrationModel):

    # Attributes:

    cdef unique_ptr[INoProbabilityCalibrationModel] probability_calibration_model_ptr


cdef inline MarginalProbabilityCalibrationModel create_marginal_probability_calibration_model(unique_ptr[IMarginalProbabilityCalibrationModel] marginal_probability_calibration_model_ptr):
    cdef IMarginalProbabilityCalibrationModel* ptr = marginal_probability_calibration_model_ptr.release()
    cdef INoProbabilityCalibrationModel* no_probability_calibration_model_ptr = dynamic_cast[NoProbabilityCalibrationModelPtr](ptr)
    cdef NoProbabilityCalibrationModel no_probability_calibration_model

    if no_probability_calibration_model_ptr != NULL:
        no_probability_calibration_model = NoProbabilityCalibrationModel.__new__(NoProbabilityCalibrationModel)
        no_probability_calibration_model.probability_calibration_model_ptr = unique_ptr[INoProbabilityCalibrationModel](no_probability_calibration_model_ptr)
        return no_probability_calibration_model
    else:
        del ptr
        raise RuntimeError('Encountered unsupported IMarginalProbabilityCalibrationModel object')
