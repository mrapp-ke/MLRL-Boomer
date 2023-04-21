from libcpp.cast cimport dynamic_cast
from libcpp.memory cimport unique_ptr


cdef extern from "common/prediction/probability_calibration_marginal.hpp" nogil:

    cdef cppclass IMarginalProbabilityCalibrationModel:
        pass


cdef extern from "common/prediction/probability_calibration_joint.hpp" nogil:

    cdef cppclass IJointProbabilityCalibrationModel:
        pass


cdef extern from "common/prediction/probability_calibration_no.hpp" nogil:

    cdef cppclass INoMarginalProbabilityCalibrationModel(IMarginalProbabilityCalibrationModel):
        pass

    unique_ptr[INoMarginalProbabilityCalibrationModel] createNoMarginalProbabilityCalibrationModel()

    cdef cppclass INoJointProbabilityCalibrationModel(IJointProbabilityCalibrationModel):
        pass

    unique_ptr[INoJointProbabilityCalibrationModel] createNoJointProbabilityCalibrationModel()


ctypedef INoMarginalProbabilityCalibrationModel* NoMarginalProbabilityCalibrationModelPtr


ctypedef INoJointProbabilityCalibrationModel* NoJointProbabilityCalibrationModelPtr


cdef extern from "common/prediction/probability_calibration_isotonic.hpp" nogil:

    cdef cppclass IIsotonicMarginalProbabilityCalibrationModel(IMarginalProbabilityCalibrationModel):
        pass

    unique_ptr[IIsotonicMarginalProbabilityCalibrationModel] createIsotonicMarginalProbabilityCalibrationModel()


ctypedef IIsotonicMarginalProbabilityCalibrationModel* IsotonicMarginalProbabilityCalibrationModelPtr


cdef class MarginalProbabilityCalibrationModel:

    # Functions:

    cdef IMarginalProbabilityCalibrationModel* get_marginal_probability_calibration_model_ptr(self)


cdef class JointProbabilityCalibrationModel:

    # Functions:

    cdef IJointProbabilityCalibrationModel* get_joint_probability_calibration_model_ptr(self)


cdef class NoMarginalProbabilityCalibrationModel(MarginalProbabilityCalibrationModel):

    # Attributes:

    cdef unique_ptr[INoMarginalProbabilityCalibrationModel] probability_calibration_model_ptr


cdef class NoJointProbabilityCalibrationModel(JointProbabilityCalibrationModel):

    # Attributes:

    cdef unique_ptr[INoJointProbabilityCalibrationModel] probability_calibration_model_ptr


cdef class IsotonicMarginalProbabilityCalibrationModel(MarginalProbabilityCalibrationModel):

    # Attributes:

    cdef unique_ptr[IIsotonicMarginalProbabilityCalibrationModel] probability_calibration_model_ptr


cdef inline MarginalProbabilityCalibrationModel create_marginal_probability_calibration_model(
        unique_ptr[IMarginalProbabilityCalibrationModel] marginal_probability_calibration_model_ptr):
    cdef IMarginalProbabilityCalibrationModel* ptr = marginal_probability_calibration_model_ptr.release()
    cdef INoMarginalProbabilityCalibrationModel* no_marginal_probability_calibration_model_ptr = \
        dynamic_cast[NoMarginalProbabilityCalibrationModelPtr](ptr)
    cdef NoMarginalProbabilityCalibrationModel no_marginal_probability_calibration_model
    cdef IIsotonicMarginalProbabilityCalibrationModel* isotonic_marginal_probability_calibration_model_ptr
    cdef IsotonicMarginalProbabilityCalibrationModel isotonic_marginal_probability_calibration_model

    if no_marginal_probability_calibration_model_ptr != NULL:
        no_marginal_probability_calibration_model = \
            NoMarginalProbabilityCalibrationModel.__new__(NoMarginalProbabilityCalibrationModel)
        no_marginal_probability_calibration_model.probability_calibration_model_ptr = \
            unique_ptr[INoMarginalProbabilityCalibrationModel](no_marginal_probability_calibration_model_ptr)
        return no_marginal_probability_calibration_model
    else:
        isotonic_marginal_probability_calibration_model_ptr = \
            dynamic_cast[IsotonicMarginalProbabilityCalibrationModelPtr](ptr)
        
        if isotonic_marginal_probability_calibration_model_ptr != NULL:
            isotonic_marginal_probability_calibration_model = \
                IsotonicMarginalProbabilityCalibrationModel.__new__(IsotonicMarginalProbabilityCalibrationModel)
            isotonic_marginal_probability_calibration_model.probability_calibration_model_ptr = \
                unique_ptr[IIsotonicMarginalProbabilityCalibrationModel](
                    isotonic_marginal_probability_calibration_model_ptr)
            return isotonic_marginal_probability_calibration_model
        else:
            del ptr
            raise RuntimeError('Encountered unsupported IMarginalProbabilityCalibrationModel object')


cdef inline JointProbabilityCalibrationModel create_joint_probability_calibration_model(
        unique_ptr[IJointProbabilityCalibrationModel] joint_probability_calibration_model_ptr):
    cdef IJointProbabilityCalibrationModel* ptr = joint_probability_calibration_model_ptr.release()
    cdef INoJointProbabilityCalibrationModel* no_joint_probability_calibration_model_ptr = \
        dynamic_cast[NoJointProbabilityCalibrationModelPtr](ptr)
    cdef NoJointProbabilityCalibrationModel no_joint_probability_calibration_model

    if no_joint_probability_calibration_model_ptr != NULL:
        no_joint_probability_calibration_model = \
            NoJointProbabilityCalibrationModel.__new__(NoJointProbabilityCalibrationModel)
        no_joint_probability_calibration_model.probability_calibration_model_ptr = \
            unique_ptr[INoJointProbabilityCalibrationModel](no_joint_probability_calibration_model_ptr)
        return no_joint_probability_calibration_model
    else:
        del ptr
        raise RuntimeError('Encountered unsupported IJointProbabilityCalibrationModel object')
