#include "mlrl/common/input/feature_vector_equal.hpp"

std::unique_ptr<IFeatureVector> EqualFeatureVector::createFilteredFeatureVector(
  std::unique_ptr<IFeatureVector>& existing, uint32 start, uint32 end, bool inverse) const {
    throw std::runtime_error("Function EqualFeatureVector::createFilteredFeatureVector should never be called");
}

std::unique_ptr<IFeatureVector> EqualFeatureVector::createFilteredFeatureVector(
  std::unique_ptr<IFeatureVector>& existing, const CoverageMask& coverageMask) const {
    throw std::runtime_error("Function EqualFeatureVector::createFilteredFeatureVector should never be called");
}
