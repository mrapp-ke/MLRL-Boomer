#include "mlrl/common/input/feature_type_nominal.hpp"

bool NominalFeatureType::isOrdinal() const {
    return false;
}

bool NominalFeatureType::isNominal() const {
    return true;
}

std::unique_ptr<IFeatureVector> NominalFeatureType::createFeatureVector(
  uint32 featureIndex, const FortranContiguousConstView<const float32>& featureMatrix) const {
    // TODO Implement
    return nullptr;
}

std::unique_ptr<IFeatureVector> NominalFeatureType::createFeatureVector(
  uint32 featureIndex, const CscConstView<const float32>& featureMatrix) const {
    // TODO Implement
    return nullptr;
}
