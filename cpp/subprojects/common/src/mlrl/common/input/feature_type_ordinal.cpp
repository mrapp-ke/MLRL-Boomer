#include "mlrl/common/input/feature_type_ordinal.hpp"

bool OrdinalFeatureType::isOrdinal() const {
    return true;
}

bool OrdinalFeatureType::isNominal() const {
    return false;
}

std::unique_ptr<IFeatureVector> OrdinalFeatureType::createFeatureVector(
  uint32 featureIndex, const FortranContiguousConstView<const float32>& featureMatrix) const {
    // TODO Implement
    return nullptr;
}

std::unique_ptr<IFeatureVector> OrdinalFeatureType::createFeatureVector(
  uint32 featureIndex, const CscConstView<const float32>& featureMatrix) const {
    // TODO Implement
    return nullptr;
}
