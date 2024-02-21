#include "mlrl/common/sampling/weight_vector_dense.hpp"

#include "mlrl/common/thresholds/feature_space.hpp"
#include "mlrl/common/thresholds/thresholds_subset.hpp"

template<typename T>
DenseWeightVector<T>::DenseWeightVector(uint32 numElements, bool init)
    : DenseVectorDecorator<AllocatedVector<T>>(AllocatedVector<T>(numElements, init)), numNonZeroWeights_(0) {}

template<typename T>
uint32 DenseWeightVector<T>::getNumNonZeroWeights() const {
    return numNonZeroWeights_;
}

template<typename T>
void DenseWeightVector<T>::setNumNonZeroWeights(uint32 numNonZeroWeights) {
    numNonZeroWeights_ = numNonZeroWeights;
}

template<typename T>
bool DenseWeightVector<T>::hasZeroWeights() const {
    return numNonZeroWeights_ < this->getNumElements();
}

template<typename T>
std::unique_ptr<IThresholdsSubset> DenseWeightVector<T>::createThresholdsSubset(IFeatureSpace& featureSpace) const {
    return featureSpace.createSubset(*this);
}

template class DenseWeightVector<uint32>;
