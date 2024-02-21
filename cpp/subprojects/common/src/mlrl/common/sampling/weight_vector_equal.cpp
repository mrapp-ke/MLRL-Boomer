#include "mlrl/common/sampling/weight_vector_equal.hpp"

#include "mlrl/common/thresholds/feature_space.hpp"
#include "mlrl/common/thresholds/thresholds_subset.hpp"

EqualWeightVector::EqualWeightVector(uint32 numElements) : numElements_(numElements) {}

uint32 EqualWeightVector::getNumElements() const {
    return numElements_;
}

uint32 EqualWeightVector::operator[](uint32 pos) const {
    return 1;
}

uint32 EqualWeightVector::getNumNonZeroWeights() const {
    return numElements_;
}

bool EqualWeightVector::hasZeroWeights() const {
    return false;
}

std::unique_ptr<IThresholdsSubset> EqualWeightVector::createThresholdsSubset(IFeatureSpace& featureSpace) const {
    return featureSpace.createSubset(*this);
}
