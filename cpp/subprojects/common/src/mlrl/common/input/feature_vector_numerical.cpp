#include "mlrl/common/input/feature_vector_numerical.hpp"

NumericalFeatureVector::NumericalFeatureVector(uint32 numElements, float32 sparseValue)
    : vector_(SparseArrayVector<float32>(numElements)), sparseValue_(sparseValue) {}

NumericalFeatureVector::iterator NumericalFeatureVector::begin() {
    return vector_.begin();
}

NumericalFeatureVector::iterator NumericalFeatureVector::end() {
    return vector_.end();
}

NumericalFeatureVector::const_iterator NumericalFeatureVector::cbegin() const {
    return vector_.cbegin();
}

NumericalFeatureVector::const_iterator NumericalFeatureVector::cend() const {
    return vector_.cend();
}

float32 NumericalFeatureVector::getSparseValue() const {
    return sparseValue_;
}

void NumericalFeatureVector::setNumElements(uint32 numElements, bool freeMemory) {
    return vector_.setNumElements(numElements, freeMemory);
}

uint32 NumericalFeatureVector::getNumElements() const {
    return vector_.getNumElements();
}

std::unique_ptr<IFeatureVector> NumericalFeatureVector::createFilteredFeatureVector(
  std::unique_ptr<IFeatureVector>& existing, uint32 start, uint32 end) const {
    // TODO Implement
    return nullptr;
}

std::unique_ptr<IFeatureVector> NumericalFeatureVector::createFilteredFeatureVector(
  std::unique_ptr<IFeatureVector>& existing, const CoverageMask& coverageMask) const {
    // TODO Implement
    return nullptr;
}
