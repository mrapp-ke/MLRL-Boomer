#include "common/prediction/probability_calibration_isotonic.hpp"

IsotonicMarginalProbabilityCalibrationModel::IsotonicMarginalProbabilityCalibrationModel(uint32 numLabels)
    : binsPerLabel_(ListOfLists<Tuple<float64>>(numLabels)) {}

float64 IsotonicMarginalProbabilityCalibrationModel::calibrateMarginalProbability(uint32 labelIndex,
                                                                                  float64 marginalProbability) const {
    // Find the bins that impose a lower and upper bound on the marginal probability...
    ListOfLists<Tuple<float64>>::const_iterator begin = binsPerLabel_.row_cbegin(labelIndex);
    ListOfLists<Tuple<float64>>::const_iterator end = binsPerLabel_.row_cend(labelIndex);
    ListOfLists<Tuple<float64>>::const_iterator it =
      std::lower_bound(begin, end, marginalProbability, [=](const Tuple<float64>& lhs, const float64& rhs) {
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
    float64 t = (marginalProbability - lowerBound.first) / (upperBound.first - lowerBound.first);
    return lowerBound.second + (t * (upperBound.second - lowerBound.second));
}

IsotonicMarginalProbabilityCalibrationModel::bin_list IsotonicMarginalProbabilityCalibrationModel::operator[](
  uint32 labelIndex) {
    return binsPerLabel_[labelIndex];
}

uint32 IsotonicMarginalProbabilityCalibrationModel::getNumLabels() const {
    return binsPerLabel_.getNumRows();
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

float64 IsotonicJointProbabilityCalibrationModel::calibrateJointProbability(uint32 labelVectorIndex,
                                                                            float64 jointProbability) const {
    // TODO Implement
    return jointProbability;
}

std::unique_ptr<IIsotonicJointProbabilityCalibrationModel> createIsotonicJointProbabilityCalibrationModel() {
    return std::make_unique<IsotonicJointProbabilityCalibrationModel>();
}
