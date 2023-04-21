#include "common/prediction/probability_calibration_isotonic.hpp"

float64 IsotonicMarginalProbabilityCalibrationModel::calibrateMarginalProbability(uint32 labelIndex,
                                                                                  float64 marginalProbability) const {
    // TODO Implement
    return marginalProbability;
}

std::unique_ptr<IIsotonicMarginalProbabilityCalibrationModel> createIsotonicMarginalProbabilityCalibrationModel() {
    return std::make_unique<IsotonicMarginalProbabilityCalibrationModel>();
}
