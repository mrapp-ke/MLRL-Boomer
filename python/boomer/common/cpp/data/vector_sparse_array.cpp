#include "vector_sparse_array.h"
#include <algorithm>
#include <cstdlib>


template<class T>
SparseArrayVector<T>::SparseArrayVector(uint32 numElements)
    : array_((Entry*) malloc(numElements * sizeof(Entry))), numElements_(numElements) {

}

template<class T>
SparseArrayVector<T>::~SparseArrayVector() {
    free(array_);
}

template<class T>
uint32 SparseArrayVector<T>::getNumElements() const {
    return numElements_;
}

template<class T>
void SparseArrayVector<T>::setNumElements(uint32 numElements) {
    if (numElements != numElements_) {
        numElements_ = numElements;
        array_ = (Entry*) realloc(array_, numElements * sizeof(Entry));
    }
}

template<class T>
typename SparseArrayVector<T>::iterator SparseArrayVector<T>::begin() {
    return array_;
}

template<class T>
typename SparseArrayVector<T>::iterator SparseArrayVector<T>::end() {
    return &array_[numElements_];
}

template<class T>
typename SparseArrayVector<T>::const_iterator SparseArrayVector<T>::cbegin() const {
    return array_;
}

template<class T>
typename SparseArrayVector<T>::const_iterator SparseArrayVector<T>::cend() const {
    return &array_[numElements_];
}

template<class T>
void SparseArrayVector<T>::sortByValues() {
    struct {

        bool operator()(const SparseArrayVector<T>::Entry& a, const SparseArrayVector<T>::Entry& b) const {
            return a.value < b.value;
        }

    } comparator;
    std::sort(this->begin(), this->end(), comparator);
}

template class SparseArrayVector<float32>;
template class SparseArrayVector<float64>;
