#include "mlrl/common/input/feature_vector_equal.hpp"

uint32 EqualFeatureVector::getNumElements() const {
    return 0;
}

std::unique_ptr<IFeatureVector> EqualFeatureVector::createFilteredFeatureVector(
  std::unique_ptr<IFeatureVector>& existing, uint32 start, uint32 end) const {
    // TODO Implement
    return nullptr;
}

std::unique_ptr<IFeatureVector> EqualFeatureVector::createFilteredFeatureVector(
  std::unique_ptr<IFeatureVector>& existing, const CoverageMask& coverageMask) const {
    // TODO Implement
    return nullptr;
}
