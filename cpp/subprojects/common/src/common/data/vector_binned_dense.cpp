#include "common/data/vector_binned_dense.hpp"


template<typename T>
DenseBinnedVector<T>::Iterator::Iterator(const DenseBinnedVector<T>& vector, uint32 index)
    : vector_(vector), index_(index) {

}

template<typename T>
typename DenseBinnedVector<T>::Iterator::reference DenseBinnedVector<T>::Iterator::operator[](uint32 index) const {
    uint32 binIndex = vector_.binIndices_[index];
    return vector_.values_[binIndex];
}

template<typename T>
typename DenseBinnedVector<T>::Iterator::reference DenseBinnedVector<T>::Iterator::operator*() const {
    uint32 binIndex = vector_.binIndices_[index_];
    return vector_.values_[binIndex];
}

template<typename T>
typename DenseBinnedVector<T>::Iterator& DenseBinnedVector<T>::Iterator::operator++() {
    ++index_;
    return *this;
}

template<typename T>
typename DenseBinnedVector<T>::Iterator& DenseBinnedVector<T>::Iterator::operator++(int n) {
    index_++;
    return *this;
}

template<typename T>
typename DenseBinnedVector<T>::Iterator& DenseBinnedVector<T>::Iterator::operator--() {
    --index_;
    return *this;
}

template<typename T>
typename DenseBinnedVector<T>::Iterator& DenseBinnedVector<T>::Iterator::operator--(int n) {
    index_--;
    return *this;
}

template<typename T>
bool DenseBinnedVector<T>::Iterator::operator!=(const DenseBinnedVector<T>::Iterator& rhs) const {
    return index_ != rhs.index_;
}

template<typename T>
bool DenseBinnedVector<T>::Iterator::operator==(const DenseBinnedVector<T>::Iterator& rhs) const {
    return index_ == rhs.index_;
}

template<typename T>
typename DenseBinnedVector<T>::Iterator::difference_type DenseBinnedVector<T>::Iterator::operator-(
        const DenseBinnedVector<T>::Iterator& rhs) const {
    return (difference_type) index_ - (difference_type) rhs.index_;
}

template<typename T>
DenseBinnedVector<T>::DenseBinnedVector(uint32 numElements, uint32 numBins)
    : binIndices_(DenseVector<uint32>(numElements)), values_(DenseVector<T>(numBins)) {

}

template<typename T>
typename DenseBinnedVector<T>::const_iterator DenseBinnedVector<T>::cbegin() const {
    return DenseBinnedVector<T>::Iterator(*this, 0);
}

template<typename T>
typename DenseBinnedVector<T>::const_iterator DenseBinnedVector<T>::cend() const {
    return DenseBinnedVector<T>::Iterator(*this, binIndices_.getNumElements());
}

template<typename T>
typename DenseBinnedVector<T>::index_binned_iterator DenseBinnedVector<T>::indices_binned_begin() {
    return binIndices_.begin();
}

template<typename T>
typename DenseBinnedVector<T>::index_binned_iterator DenseBinnedVector<T>::indices_binned_end() {
    return binIndices_.end();
}

template<typename T>
typename DenseBinnedVector<T>::index_binned_const_iterator DenseBinnedVector<T>::indices_binned_cbegin() const {
    return binIndices_.cbegin();
}

template<typename T>
typename DenseBinnedVector<T>::index_binned_const_iterator DenseBinnedVector<T>::indices_binned_cend() const {
    return binIndices_.cend();
}

template<typename T>
typename DenseBinnedVector<T>::binned_iterator DenseBinnedVector<T>::binned_begin() {
    return values_.begin();
}

template<typename T>
typename DenseBinnedVector<T>::binned_iterator DenseBinnedVector<T>::binned_end() {
    return values_.end();
}

template<typename T>
typename DenseBinnedVector<T>::binned_const_iterator DenseBinnedVector<T>::binned_cbegin() const {
    return values_.cbegin();
}

template<typename T>
typename DenseBinnedVector<T>::binned_const_iterator DenseBinnedVector<T>::binned_cend() const {
    return values_.cend();
}

template<typename T>
uint32 DenseBinnedVector<T>::getNumElements() const {
    return binIndices_.getNumElements();
}

template<typename T>
uint32 DenseBinnedVector<T>::getNumBins() const {
    return values_.getNumElements();
}

template<typename T>
void DenseBinnedVector<T>::setNumBins(uint32 numBins, bool freeMemory) {
    values_.setNumElements(numBins, freeMemory);
}

template class DenseBinnedVector<uint8>;
template class DenseBinnedVector<uint32>;
template class DenseBinnedVector<float32>;
template class DenseBinnedVector<float64>;
