#include "common/data/vector_sparse_unordered.hpp"
#include "common/data/arrays.hpp"
#include "common/data/tuple.hpp"
#include "common/data/triple.hpp"


template<typename T>
SparseUnorderedVector<T>::SparseUnorderedVector(uint32 maxElements)
    : ptrs_(new IndexedValue<T>*[maxElements] {}), maxElements_(maxElements) {

}

template<typename T>
SparseUnorderedVector<T>::~SparseUnorderedVector() {
    delete[] ptrs_;
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
uint32 SparseUnorderedVector<T>::getMaxElements() const {
    return maxElements_;
}

template<typename T>
IndexedValue<T>& SparseUnorderedVector<T>::operator[](uint32 index) {
    IndexedValue<T>* ptr = ptrs_[index];

    if (!ptr) {
        values_.emplace_back(index);
        ptr = &values_.back();
        ptrs_[index] = ptr;
    }

    return *ptr;
}

template<typename T>
void SparseUnorderedVector<T>::erase(uint32 index) {
    IndexedValue<T>* ptr = ptrs_[index];

    if (ptr) {
        const IndexedValue<T>& lastEntry = values_.back();
        uint32 lastIndex = lastEntry.index;

        if (lastIndex != index) {
            *ptr = lastEntry;
            ptrs_[lastIndex] = ptr;
        }

        ptrs_[index] = nullptr;
        values_.resize(values_.size() - 1);
    }
}

template<typename T>
void SparseUnorderedVector<T>::clear() {
    setArrayToValue<IndexedValue<T>*>(ptrs_, maxElements_, nullptr);
    values_.clear();
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
