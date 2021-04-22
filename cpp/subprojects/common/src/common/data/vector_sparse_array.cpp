#include "common/data/vector_sparse_array.hpp"
#include <algorithm>


template<class T>
SparseArrayVector<T>::SparseArrayVector(uint32 numElements)
    : vector_(DenseVector<IndexedValue<T>>(numElements)) {

}

template<class T>
typename SparseArrayVector<T>::iterator SparseArrayVector<T>::begin() {
    return vector_.begin();
}

template<class T>
typename SparseArrayVector<T>::iterator SparseArrayVector<T>::end() {
    return vector_.end();
}

template<class T>
typename SparseArrayVector<T>::const_iterator SparseArrayVector<T>::cbegin() const {
    return vector_.cbegin();
}

template<class T>
typename SparseArrayVector<T>::const_iterator SparseArrayVector<T>::cend() const {
    return vector_.cend();
}

template<class T>
uint32 SparseArrayVector<T>::getNumElements() const {
    return vector_.getNumElements();
}

template<class T>
void SparseArrayVector<T>::setNumElements(uint32 numElements, bool freeMemory) {
    vector_.setNumElements(numElements, freeMemory);
}

template<class T>
void SparseArrayVector<T>::sortByValues() {
    std::sort(vector_.begin(), vector_.end(), [=](const IndexedValue<T>& a, const IndexedValue<T>& b) {
        return a.value < b.value;
    });
}

template class SparseArrayVector<uint8>;
template class SparseArrayVector<uint32>;
template class SparseArrayVector<float32>;
template class SparseArrayVector<float64>;
