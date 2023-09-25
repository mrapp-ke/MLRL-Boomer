#include "mlrl/common/input/feature_type_numerical.hpp"

bool NumericalFeatureType::isOrdinal() const {
    return false;
}

bool NumericalFeatureType::isNominal() const {
    return false;
}

std::unique_ptr<IFeatureVector> NumericalFeatureType::createFeatureVector(
  uint32 featureIndex, const FortranContiguousConstView<const float32>& featureMatrix) const {
    // TODO Implement
    return nullptr;
}

std::unique_ptr<IFeatureVector> NumericalFeatureType::createFeatureVector(
  uint32 featureIndex, const CscConstView<const float32>& featureMatrix) const {
    // TODO Implement
    return nullptr;
}
