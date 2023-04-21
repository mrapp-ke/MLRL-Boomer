#include "common/prediction/probability_calibration_isotonic.hpp"

float64 IsotonicMarginalProbabilityCalibrationModel::calibrateMarginalProbability(uint32 labelIndex,
                                                                                  float64 marginalProbability) const {
    // TODO Implement
    return marginalProbability;
}

std::unique_ptr<IIsotonicMarginalProbabilityCalibrationModel> createIsotonicMarginalProbabilityCalibrationModel() {
    return std::make_unique<IsotonicMarginalProbabilityCalibrationModel>();
}

float64 IsotonicJointProbabilityCalibrationModel::calibrateJointProbability(float64 jointProbability) const {
    // TODO Implement
    return jointProbability;
}

std::unique_ptr<IIsotonicJointProbabilityCalibrationModel> createIsotonicJointProbabilityCalibrationModel() {
    return std::make_unique<IsotonicJointProbabilityCalibrationModel>();
}
