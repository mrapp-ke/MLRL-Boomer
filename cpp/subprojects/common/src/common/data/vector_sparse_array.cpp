#include "common/data/vector_sparse_array.hpp"
#include <algorithm>
#include <cstdlib>


template<class T>
SparseArrayVector<T>::SparseArrayVector(uint32 numElements)
    : array_((IndexedValue<T>*) malloc(numElements * sizeof(IndexedValue<T>))), numElements_(numElements),
      maxCapacity_(numElements) {

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
void SparseArrayVector<T>::setNumElements(uint32 numElements, bool freeMemory) {
    if (numElements < maxCapacity_) {
        if (freeMemory) {
            array_ = (IndexedValue<T>*) realloc(array_, numElements * sizeof(IndexedValue<T>));
            maxCapacity_ = numElements;
        }
    } else if (numElements > maxCapacity_) {
        array_ = (IndexedValue<T>*) realloc(array_, numElements * sizeof(IndexedValue<T>));
        maxCapacity_ = numElements;
    }

    numElements_ = numElements;
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

        bool operator()(const IndexedValue<T>& a, const IndexedValue<T>& b) const {
            return a.value < b.value;
        }

    } comparator;
    std::sort(this->begin(), this->end(), comparator);
}

template class SparseArrayVector<float32>;
template class SparseArrayVector<float64>;
