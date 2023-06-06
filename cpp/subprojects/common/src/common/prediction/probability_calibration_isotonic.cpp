#include "common/prediction/probability_calibration_isotonic.hpp"

static inline float64 calibrateProbability(const AbstractIsotonicProbabilityCalibrationModel::const_bin_list bins,
                                           float64 probability) {
    // Find the bins that impose a lower and upper bound on the probability...
    ListOfLists<Tuple<float64>>::const_iterator begin = bins.cbegin();
    ListOfLists<Tuple<float64>>::const_iterator end = bins.cend();
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

AbstractIsotonicProbabilityCalibrationModel::AbstractIsotonicProbabilityCalibrationModel(uint32 numLists)
    : binsPerList_(ListOfLists<Tuple<float64>>(numLists)) {}

AbstractIsotonicProbabilityCalibrationModel::bin_list AbstractIsotonicProbabilityCalibrationModel::operator[](
  uint32 listIndex) {
    return binsPerList_[listIndex];
}

AbstractIsotonicProbabilityCalibrationModel::const_bin_list AbstractIsotonicProbabilityCalibrationModel::operator[](
  uint32 listIndex) const {
    return binsPerList_[listIndex];
}

uint32 AbstractIsotonicProbabilityCalibrationModel::getNumBinLists() const {
    return binsPerList_.getNumRows();
}

void AbstractIsotonicProbabilityCalibrationModel::addBin(uint32 listIndex, float64 threshold, float64 probability) {
    ListOfLists<Tuple<float64>>::row row = binsPerList_[listIndex];
    row.emplace_back(threshold, probability);
}

void AbstractIsotonicProbabilityCalibrationModel::visit(BinVisitor visitor) const {
    uint32 numLists = binsPerList_.getNumRows();

    for (uint32 i = 0; i < numLists; i++) {
        ListOfLists<Tuple<float64>>::const_row bins = binsPerList_[i];

        for (auto it = bins.cbegin(); it != bins.cend(); it++) {
            const Tuple<float64>& bin = *it;
            visitor(i, bin.first, bin.second);
        }
    }
}

IsotonicMarginalProbabilityCalibrationModel::IsotonicMarginalProbabilityCalibrationModel(uint32 numLabels)
    : AbstractIsotonicProbabilityCalibrationModel(numLabels) {}

float64 IsotonicMarginalProbabilityCalibrationModel::calibrateMarginalProbability(uint32 labelIndex,
                                                                                  float64 marginalProbability) const {
    return calibrateProbability((*this)[labelIndex], marginalProbability);
}

std::unique_ptr<IIsotonicMarginalProbabilityCalibrationModel> createIsotonicMarginalProbabilityCalibrationModel(
  uint32 numLabels) {
    return std::make_unique<IsotonicMarginalProbabilityCalibrationModel>(numLabels);
}

IsotonicJointProbabilityCalibrationModel::IsotonicJointProbabilityCalibrationModel(uint32 numLabelVectors)
    : AbstractIsotonicProbabilityCalibrationModel(numLabelVectors) {}

float64 IsotonicJointProbabilityCalibrationModel::calibrateJointProbability(uint32 labelVectorIndex,
                                                                            float64 jointProbability) const {
    return calibrateProbability((*this)[labelVectorIndex], jointProbability);
}

std::unique_ptr<IIsotonicJointProbabilityCalibrationModel> createIsotonicJointProbabilityCalibrationModel(
  uint32 numLabelVectors) {
    return std::make_unique<IsotonicJointProbabilityCalibrationModel>(numLabelVectors);
}
