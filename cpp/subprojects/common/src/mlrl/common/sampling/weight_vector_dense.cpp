#include "mlrl/common/sampling/weight_vector_dense.hpp"

#include "mlrl/common/thresholds/thresholds.hpp"
#include "mlrl/common/thresholds/thresholds_subset.hpp"

template<typename T>
DenseWeightVector<T>::DenseWeightVector(uint32 numElements, bool init)
    : WritableVectorDecorator<AllocatedView<Vector<T>>>(AllocatedView<Vector<T>>(numElements, init)),
      numNonZeroWeights_(0) {}

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
    return numNonZeroWeights_ < this->view_.numElements;
}

template<typename T>
std::unique_ptr<IThresholdsSubset> DenseWeightVector<T>::createThresholdsSubset(IThresholds& thresholds) const {
    return thresholds.createSubset(*this);
}

template class DenseWeightVector<uint32>;
