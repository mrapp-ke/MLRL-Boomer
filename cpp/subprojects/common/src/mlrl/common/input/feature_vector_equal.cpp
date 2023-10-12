#include "mlrl/common/input/feature_vector_equal.hpp"

static inline std::unique_ptr<IFeatureVector> createFilteredFeatureVectorInternally(
  std::unique_ptr<IFeatureVector>& existing) {
    if (dynamic_cast<EqualFeatureVector*>(existing.get()) != nullptr) {
        return std::move(existing);
    }

    return std::make_unique<EqualFeatureVector>();
}

uint32 EqualFeatureVector::getNumElements() const {
    return 0;
}

std::unique_ptr<IFeatureVector> EqualFeatureVector::createFilteredFeatureVector(
  std::unique_ptr<IFeatureVector>& existing, uint32 start, uint32 end) const {
    return createFilteredFeatureVectorInternally(existing);
}

std::unique_ptr<IFeatureVector> EqualFeatureVector::createFilteredFeatureVector(
  std::unique_ptr<IFeatureVector>& existing, const CoverageMask& coverageMask) const {
    return createFilteredFeatureVectorInternally(existing);
}
