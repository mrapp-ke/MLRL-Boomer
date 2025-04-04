#include "mlrl/common/sampling/weight_vector_dense.hpp"

#include "mlrl/common/rule_refinement/feature_space.hpp"
#include "mlrl/common/rule_refinement/feature_subspace.hpp"

template<typename T>
DenseWeightVector<T>::DenseWeightVector(uint32 numElements, bool init)
    : ClearableViewDecorator<DenseVectorDecorator<AllocatedVector<T>>>(AllocatedVector<T>(numElements, init)),
      numNonZeroWeights_(0) {}

template<typename T>
void DenseWeightVector<T>::set(uint32 pos, DenseWeightVector<T>::weight_type weight) {
    (*this)[pos] = weight;
}

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
std::unique_ptr<IFeatureSubspace> DenseWeightVector<T>::createFeatureSubspace(IFeatureSpace& featureSpace) const {
    return featureSpace.createSubspace(*this);
}

template class DenseWeightVector<uint16>;
template class DenseWeightVector<float32>;
