#include "vector_mapping_dense.h"
#include <cstdlib>


template<class T>
static inline void initEntries(T** array, uint32 start, uint32 end) {
    for (uint32 i = start; i < end; i++) {
        array[i] = new T();
    }
}

template<class T>
static inline void deleteEntries(T** array, uint32 start, uint32 end) {
    for (uint32 i = start; i < end; i++) {
        delete array[i];
    }
}

template<class T>
static inline void clearEntries(T** array, uint32 start, uint32 end) {
    for (uint32 i = start; i < end; i++) {
        T& entry = *array[i];
        entry.clear();
    }
}

template<class T>
DenseMappingVector<T>::Iterator::Iterator(const DenseMappingVector<T>& vector, uint32 index)
    : vector_(vector), index_(index) {

}

template<class T>
typename DenseMappingVector<T>::Entry& DenseMappingVector<T>::Iterator::operator[](uint32 index) {
    return *vector_.array_[index];
}

template<class T>
typename DenseMappingVector<T>::Entry& DenseMappingVector<T>::Iterator::operator*() {
    return *vector_.array_[index_];
}

template<class T>
typename DenseMappingVector<T>::Iterator& DenseMappingVector<T>::Iterator::operator++(int n) {
    index_++;
    return *this;
}

template<class T>
bool DenseMappingVector<T>::Iterator::operator!=(const Iterator& rhs) const {
    return index_ != rhs.index_;
}

template<class T>
DenseMappingVector<T>::ConstIterator::ConstIterator(const DenseMappingVector<T>& vector, uint32 index)
    : vector_(vector), index_(index) {

}

template<class T>
const typename DenseMappingVector<T>::Entry& DenseMappingVector<T>::ConstIterator::operator[](uint32 index) const {
    return *vector_.array_[index];
}

template<class T>
const typename DenseMappingVector<T>::Entry& DenseMappingVector<T>::ConstIterator::operator*() const {
    return *vector_.array_[index_];
}

template<class T>
typename DenseMappingVector<T>::ConstIterator& DenseMappingVector<T>::ConstIterator::operator++(int n) {
    index_++;
    return *this;
}

template<class T>
bool DenseMappingVector<T>::ConstIterator::operator!=(const ConstIterator& rhs) const {
    return index_ != rhs.index_;
}

template<class T>
DenseMappingVector<T>::DenseMappingVector(uint32 numElements)
    : array_((Entry**) malloc(numElements * sizeof(Entry*))), numElements_(numElements),
      maxCapacity_(numElements) {
    initEntries<Entry>(array_, 0, numElements);
}

template<class T>
DenseMappingVector<T>::~DenseMappingVector() {
    deleteEntries<Entry>(array_, 0, numElements_);
    free(array_);
}

template<class T>
typename DenseMappingVector<T>::iterator DenseMappingVector<T>::begin() {
    return Iterator(*this, 0);
}

template<class T>
typename DenseMappingVector<T>::iterator DenseMappingVector<T>::end() {
    return Iterator(*this, numElements_);
}

template<class T>
typename DenseMappingVector<T>::const_iterator DenseMappingVector<T>::cbegin() const {
    return ConstIterator(*this, 0);
}

template<class T>
typename DenseMappingVector<T>::const_iterator DenseMappingVector<T>::cend() const {
    return ConstIterator(*this, numElements_);
}

template<class T>
uint32 DenseMappingVector<T>::getNumElements() const {
    return numElements_;
}

template<class T>
void DenseMappingVector<T>::setNumElements(uint32 numElements, bool freeMemory) {
    if (numElements < maxCapacity_) {
        if (freeMemory) {
            deleteEntries<Entry>(array_, numElements, maxCapacity_);
            array_ = (Entry**) realloc(array_, numElements * sizeof(Entry*));
            maxCapacity_ = numElements;
        } else {
            clearEntries<Entry>(array_, numElements, maxCapacity_);
        }
    } else if (numElements > maxCapacity_) {
        array_ = (Entry**) realloc(array_, numElements * sizeof(Entry*));
        initEntries<Entry>(array_, maxCapacity_, numElements);
        maxCapacity_ = numElements;
    }

    numElements_ = numElements;
}

template<class T>
typename DenseMappingVector<T>::Entry& DenseMappingVector<T>::getEntry(uint32 pos) {
    return *array_[pos];
}

template<class T>
void DenseMappingVector<T>::clear() {
    clearEntries<Entry>(array_, 0, numElements_);
}

template class DenseMappingVector<uint32>;
