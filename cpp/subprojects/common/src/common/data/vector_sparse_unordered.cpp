#include "common/data/vector_sparse_unordered.hpp"
#include "common/data/arrays.hpp"
#include "common/data/tuple.hpp"
#include "common/data/triple.hpp"
#include <limits>


static const uint32 MAX_INDEX = std::numeric_limits<uint32>::max();

template<typename T>
SparseUnorderedVector<T>::SparseUnorderedVector(uint32 maxElements)
    : indices_(new uint32[maxElements]) {
    setArrayToValue(indices_, maxElements, MAX_INDEX);
}

template<typename T>
SparseUnorderedVector<T>::~SparseUnorderedVector() {
    delete[] indices_;
}

template<typename T>
typename SparseUnorderedVector<T>::iterator SparseUnorderedVector<T>::begin() {
    return values_.begin();
}

template<typename T>
typename SparseUnorderedVector<T>::iterator SparseUnorderedVector<T>::end() {
    return values_.end();
}

template<typename T>
typename SparseUnorderedVector<T>::const_iterator SparseUnorderedVector<T>::cbegin() const {
    return values_.cbegin();
}

template<typename T>
typename SparseUnorderedVector<T>::const_iterator SparseUnorderedVector<T>::cend() const {
    return values_.cend();
}

template<typename T>
uint32 SparseUnorderedVector<T>::getNumElements() const {
    return (uint32) values_.size();
}

template<typename T>
const IndexedValue<T>* SparseUnorderedVector<T>::operator[](uint32 index) const {
    uint32 i = indices_[index];
    return i == MAX_INDEX ? nullptr : &values_[i];
}

template<typename T>
IndexedValue<T>& SparseUnorderedVector<T>::emplace(uint32 index) {
    uint32 i = indices_[index];

    if (i == MAX_INDEX) {
        indices_[index] = (uint32) values_.size();
        values_.emplace_back(index);
        return values_.back();
    }

    return values_[i];
}

template<typename T>
IndexedValue<T>& SparseUnorderedVector<T>::emplace(uint32 index, const T& defaultValue) {
    uint32 i = indices_[index];

    if (i == MAX_INDEX) {
        indices_[index] = (uint32) values_.size();
        values_.emplace_back(index, defaultValue);
        return values_.back();
    }

    return values_[i];
}

template<typename T>
void SparseUnorderedVector<T>::erase(uint32 index) {
    uint32 i = indices_[index];

    if (i != MAX_INDEX) {
        const IndexedValue<T>& lastEntry = values_.back();
        uint32 lastIndex = lastEntry.index;

        if (lastIndex != index) {
            values_[i] = lastEntry;
            indices_[lastIndex] = i;
        }

        indices_[index] = MAX_INDEX;
        values_.pop_back();
    }
}

template<typename T>
void SparseUnorderedVector<T>::clear() {
    while (!values_.empty()) {
        const IndexedValue<T>& lastEntry = values_.back();
        indices_[lastEntry.index] = MAX_INDEX;
        values_.pop_back();
    }
}

template class SparseUnorderedVector<uint8>;
template class SparseUnorderedVector<uint32>;
template class SparseUnorderedVector<float32>;
template class SparseUnorderedVector<float64>;
template class SparseUnorderedVector<Tuple<uint8>>;
template class SparseUnorderedVector<Tuple<uint32>>;
template class SparseUnorderedVector<Tuple<float32>>;
template class SparseUnorderedVector<Tuple<float64>>;
template class SparseUnorderedVector<Triple<uint8>>;
template class SparseUnorderedVector<Triple<uint32>>;
template class SparseUnorderedVector<Triple<float32>>;
template class SparseUnorderedVector<Triple<float64>>;
