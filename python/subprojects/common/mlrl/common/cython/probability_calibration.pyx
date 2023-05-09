"""
@author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from libcpp.utility cimport move

SERIALIZATION_VERVSION = 0


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
    
    cdef __serialize_bin(self, uint32 label_index, float64 threshold, float64 probability):
        if len(self.state) <= label_index:
            self.state.append([])

        cdef list bin_list = self.state[label_index]
        bin_list.append((threshold, probability))

    def __reduce(self):
        self.state = []
        self.probability_calibration_model_ptr.get().visit(
            wrapBinVisitor(<void*>self, <BinCythonVisitor>self.__serialize_bin))
        cdef object state = (SERIALIZATION_VERVSION, self.state)
        self.state = None
        return (IsotonicMarginalProbabilityCalibrationModel, (), state)

    def __setstate(self, state):
        cdef int version = state[0]

        if version != SERIALIZATION_VERVSION:
            raise AssertionError('Version of the serialized IsotonicMarginalProbabilityCalibrationModel is '
                                 + str(version) + ', expected ' + str(SERIALIZATION_VERVSION))
        
        cdef list bins_per_label = state[1]
        cdef uint32 num_labels = len(bins_per_label)
        cdef unique_ptr[IIsotonicMarginalProbabilityCalibrationModel] marginal_probability_calibration_model_ptr = \
            createIsotonicMarginalProbabilityCalibrationModel(num_labels)
        cdef list bin_list
        cdef threshold, probability
        cdef uint32 i, j, num_bins

        for i in range(num_labels):
            bin_list = bins_per_label[i]
            num_bins = len(bin_list)

            for j in range(num_bins):
                threshold = bin_list[j][0]
                probability = bin_list[j][1]
                marginal_probability_calibration_model_ptr.get().addBin(i, threshold, probability)

        self.probability_calibration_model_ptr = move(marginal_probability_calibration_model_ptr)


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
