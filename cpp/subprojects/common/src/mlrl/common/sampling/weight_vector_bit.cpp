#include "mlrl/common/sampling/weight_vector_bit.hpp"

#include "mlrl/common/rule_refinement/feature_space.hpp"
#include "mlrl/common/rule_refinement/feature_subspace.hpp"

BitWeightVector::BitWeightVector(uint32 numElements, bool init) : vector_(numElements, init), numNonZeroWeights_(0) {}

uint32 BitWeightVector::getNumElements() const {
    return vector_.getNumElements();
}

uint32 BitWeightVector::getNumNonZeroWeights() const {
    return numNonZeroWeights_;
}

void BitWeightVector::setNumNonZeroWeights(uint32 numNonZeroWeights) {
    numNonZeroWeights_ = numNonZeroWeights;
}

bool BitWeightVector::hasZeroWeights() const {
    return numNonZeroWeights_ < vector_.getNumElements();
}

BitWeightVector::weight_type BitWeightVector::operator[](uint32 pos) const {
    return vector_[pos];
}

void BitWeightVector::set(uint32 pos, bool weight) {
    vector_.set(pos, weight);
}

void BitWeightVector::clear() {
    vector_.clear();
}

std::unique_ptr<IFeatureSubspace> BitWeightVector::createFeatureSubspace(IFeatureSpace& featureSpace) const {
    return featureSpace.createSubspace(*this);
}
