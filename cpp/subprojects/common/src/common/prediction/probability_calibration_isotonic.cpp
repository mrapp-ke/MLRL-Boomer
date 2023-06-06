#include "common/prediction/probability_calibration_isotonic.hpp"

static inline float64 calibrateProbability(const ListOfLists<Tuple<float64>>& bins, uint32 index, float64 probability) {
    // Find the bins that impose a lower and upper bound on the probability...
    ListOfLists<Tuple<float64>>::const_iterator begin = bins.row_cbegin(index);
    ListOfLists<Tuple<float64>>::const_iterator end = bins.row_cend(index);
    ListOfLists<Tuple<float64>>::const_iterator it =
      std::lower_bound(begin, end, probability, [=](const Tuple<float64>& lhs, const float64& rhs) {
          return lhs.first < rhs;
      });
    uint32 offset = it - begin;
    Tuple<float64> lowerBound;
    Tuple<float64> upperBound;

    if (it == end) {
        lowerBound = begin[offset - 1];
        upperBound = 1;
    } else {
        if (offset > 0) {
            lowerBound = begin[offset - 1];
        } else {
            lowerBound = 0;
        }

        upperBound = *it;
    }

    // Interpolate linearly between the probabilities associated with the lower and upper bound...
    float64 t = (probability - lowerBound.first) / (upperBound.first - lowerBound.first);
    return lowerBound.second + (t * (upperBound.second - lowerBound.second));
}

IsotonicMarginalProbabilityCalibrationModel::IsotonicMarginalProbabilityCalibrationModel(uint32 numLabels)
    : binsPerLabel_(ListOfLists<Tuple<float64>>(numLabels)) {}

float64 IsotonicMarginalProbabilityCalibrationModel::calibrateMarginalProbability(uint32 labelIndex,
                                                                                  float64 marginalProbability) const {
    return calibrateProbability(binsPerLabel_, labelIndex, marginalProbability);
}

IsotonicMarginalProbabilityCalibrationModel::bin_list IsotonicMarginalProbabilityCalibrationModel::operator[](
  uint32 listIndex) {
    return binsPerLabel_[listIndex];
}

uint32 IsotonicMarginalProbabilityCalibrationModel::getNumBinLists() const {
    return binsPerLabel_.getNumRows();
}

void IsotonicMarginalProbabilityCalibrationModel::addBin(uint32 listIndex, float64 threshold, float64 probability) {
    ListOfLists<Tuple<float64>>::row row = binsPerLabel_[listIndex];
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

IsotonicJointProbabilityCalibrationModel::IsotonicJointProbabilityCalibrationModel(uint32 numLabelVectors)
    : binsPerLabelVector_(ListOfLists<Tuple<float64>>(numLabelVectors)) {}

float64 IsotonicJointProbabilityCalibrationModel::calibrateJointProbability(uint32 labelVectorIndex,
                                                                            float64 jointProbability) const {
    return calibrateProbability(binsPerLabelVector_, labelVectorIndex, jointProbability);
}

IsotonicJointProbabilityCalibrationModel::bin_list IsotonicJointProbabilityCalibrationModel::operator[](
  uint32 listIndex) {
    return binsPerLabelVector_[listIndex];
}

uint32 IsotonicJointProbabilityCalibrationModel::getNumBinLists() const {
    return binsPerLabelVector_.getNumRows();
}

void IsotonicJointProbabilityCalibrationModel::addBin(uint32 listIndex, float64 threshold, float64 probability) {
    ListOfLists<Tuple<float64>>::row row = binsPerLabelVector_[listIndex];
    row.emplace_back(threshold, probability);
}

void IsotonicJointProbabilityCalibrationModel::visit(BinVisitor visitor) const {
    uint32 numLabelVectors = binsPerLabelVector_.getNumRows();

    for (uint32 i = 0; i < numLabelVectors; i++) {
        ListOfLists<Tuple<float64>>::const_row bins = binsPerLabelVector_[i];

        for (auto it = bins.cbegin(); it != bins.cend(); it++) {
            const Tuple<float64>& bin = *it;
            visitor(i, bin.first, bin.second);
        }
    }
}

std::unique_ptr<IIsotonicJointProbabilityCalibrationModel> createIsotonicJointProbabilityCalibrationModel(
  uint32 numLabelVectors) {
    return std::make_unique<IsotonicJointProbabilityCalibrationModel>(numLabelVectors);
}
