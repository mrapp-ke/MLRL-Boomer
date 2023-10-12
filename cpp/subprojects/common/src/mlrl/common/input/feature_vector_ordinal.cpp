#include "mlrl/common/input/feature_vector_ordinal.hpp"

OrdinalFeatureVector::OrdinalFeatureVector(uint32 numValues, uint32 numExamples, int32 majorityValue)
    : NominalFeatureVector(numValues, numExamples, majorityValue) {}

std::unique_ptr<IFeatureVector> OrdinalFeatureVector::createFilteredFeatureVector(
  std::unique_ptr<IFeatureVector>& existing, uint32 start, uint32 end) const {
    // TODO Implement
    return nullptr;
}

std::unique_ptr<IFeatureVector> OrdinalFeatureVector::createFilteredFeatureVector(
  std::unique_ptr<IFeatureVector>& existing, const CoverageMask& coverageMask) const {
    // TODO Implement
    return nullptr;
}
