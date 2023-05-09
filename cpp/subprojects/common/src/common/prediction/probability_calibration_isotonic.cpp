#include "common/prediction/probability_calibration_isotonic.hpp"

IsotonicMarginalProbabilityCalibrationModel::IsotonicMarginalProbabilityCalibrationModel(uint32 numLabels)
    : binsPerLabel_(ListOfLists<Tuple<float64>>(numLabels)) {}

float64 IsotonicMarginalProbabilityCalibrationModel::calibrateMarginalProbability(uint32 labelIndex,
                                                                                  float64 marginalProbability) const {
    // TODO Implement
    return marginalProbability;
}

IsotonicMarginalProbabilityCalibrationModel::bin_list IsotonicMarginalProbabilityCalibrationModel::operator[](
  uint32 labelIndex) {
    return binsPerLabel_[labelIndex];
}

void IsotonicMarginalProbabilityCalibrationModel::addBin(uint32 labelIndex, float64 threshold, float64 probability) {
    ListOfLists<Tuple<float64>>::row row = binsPerLabel_[labelIndex];
    row.emplace_back(threshold, probability);
}

void IsotonicMarginalProbabilityCalibrationModel::visit(BinVisitor visitor) const {
    uint32 numLabels = binsPerLabel_.getNumRows();

    for (uint32 i = 0; i < numLabels; i++) {
        ListOfLists<Tuple<float64>>::const_row bins = binsPerLabel_[i];

        for (auto it = bins.cbegin(); it != bins.cend(); it++) {
            const Tuple<float64>& bin = *it;
            visitor(i, bin.first, bin.second);
        }
    }
}

std::unique_ptr<IIsotonicMarginalProbabilityCalibrationModel> createIsotonicMarginalProbabilityCalibrationModel(
  uint32 numLabels) {
    return std::make_unique<IsotonicMarginalProbabilityCalibrationModel>(numLabels);
}

float64 IsotonicJointProbabilityCalibrationModel::calibrateJointProbability(float64 jointProbability) const {
    // TODO Implement
    return jointProbability;
}

std::unique_ptr<IIsotonicJointProbabilityCalibrationModel> createIsotonicJointProbabilityCalibrationModel() {
    return std::make_unique<IsotonicJointProbabilityCalibrationModel>();
}
