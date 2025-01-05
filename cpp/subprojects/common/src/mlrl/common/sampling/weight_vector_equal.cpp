#include "mlrl/common/sampling/weight_vector_equal.hpp"

#include "mlrl/common/rule_refinement/feature_space.hpp"
#include "mlrl/common/rule_refinement/feature_subspace.hpp"

EqualWeightVector::EqualWeightVector(uint32 numElements) : numElements_(numElements) {}

uint32 EqualWeightVector::getNumElements() const {
    return numElements_;
}

EqualWeightVector::weight_type EqualWeightVector::operator[](uint32 pos) const {
    return 1;
}

uint32 EqualWeightVector::getNumNonZeroWeights() const {
    return numElements_;
}

bool EqualWeightVector::hasZeroWeights() const {
    return false;
}

std::unique_ptr<IFeatureSubspace> EqualWeightVector::createFeatureSubspace(IFeatureSpace& featureSpace) const {
    return featureSpace.createSubspace(*this);
}
