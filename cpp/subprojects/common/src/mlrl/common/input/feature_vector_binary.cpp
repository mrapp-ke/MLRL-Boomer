#include "mlrl/common/input/feature_vector_binary.hpp"

BinaryFeatureVector::BinaryFeatureVector(uint32 numMinorityExamples, int32 minorityValue, int32 majorityValue)
    : NominalFeatureVector(1, numMinorityExamples, majorityValue) {
    this->values_begin()[0] = minorityValue;
    this->indptr_begin()[0] = 0;
}

std::unique_ptr<IFeatureVector> BinaryFeatureVector::createFilteredFeatureVector(
  std::unique_ptr<IFeatureVector>& existing, uint32 start, uint32 end) const {
    // TODO Implement
    return nullptr;
}

std::unique_ptr<IFeatureVector> BinaryFeatureVector::createFilteredFeatureVector(
  std::unique_ptr<IFeatureVector>& existing, const CoverageMask& coverageMask) const {
    // TODO Implement
    return nullptr;
}
