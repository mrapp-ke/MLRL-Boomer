from libcpp.cast cimport dynamic_cast
from libcpp.memory cimport unique_ptr


cdef extern from "common/prediction/probability_calibration.hpp" nogil:

    cdef cppclass IMarginalProbabilityCalibrationModel:
        pass


cdef extern from "common/prediction/probability_calibration_no.hpp" nogil:

    cdef cppclass INoMarginalProbabilityCalibrationModel(IMarginalProbabilityCalibrationModel):
        pass


    unique_ptr[INoMarginalProbabilityCalibrationModel] createNoMarginalProbabilityCalibrationModel()


ctypedef INoMarginalProbabilityCalibrationModel* NoMarginalProbabilityCalibrationModelPtr


cdef class MarginalProbabilityCalibrationModel:

    # Functions:

    cdef IMarginalProbabilityCalibrationModel* get_marginal_probability_calibration_model_ptr(self)


cdef class NoMarginalProbabilityCalibrationModel(MarginalProbabilityCalibrationModel):

    # Attributes:

    cdef unique_ptr[INoMarginalProbabilityCalibrationModel] probability_calibration_model_ptr


cdef inline MarginalProbabilityCalibrationModel create_marginal_probability_calibration_model(unique_ptr[IMarginalProbabilityCalibrationModel] marginal_probability_calibration_model_ptr):
    cdef IMarginalProbabilityCalibrationModel* ptr = marginal_probability_calibration_model_ptr.release()
    cdef INoMarginalProbabilityCalibrationModel* no_marginal_probability_calibration_model_ptr = dynamic_cast[NoMarginalProbabilityCalibrationModelPtr](ptr)
    cdef NoMarginalProbabilityCalibrationModel no_marginal_probability_calibration_model

    if no_marginal_probability_calibration_model_ptr != NULL:
        no_marginal_probability_calibration_model = NoMarginalProbabilityCalibrationModel.__new__(NoMarginalProbabilityCalibrationModel)
        no_marginal_probability_calibration_model.probability_calibration_model_ptr = unique_ptr[INoMarginalProbabilityCalibrationModel](no_marginal_probability_calibration_model_ptr)
        return no_marginal_probability_calibration_model
    else:
        del ptr
        raise RuntimeError('Encountered unsupported IMarginalProbabilityCalibrationModel object')
