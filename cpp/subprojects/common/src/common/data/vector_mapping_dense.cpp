#include "common/data/vector_mapping_dense.hpp"
#include "common/data/indexed_value.hpp"
#include <cstdlib>


template<class T>
static inline void initEntries(T** array, uint32 start, uint32 end) {
    for (uint32 i = start; i < end; i++) {
        array[i] = nullptr;
    }
}

template<class T>
static inline void deleteEntries(T** array, uint32 start, uint32 end) {
    for (uint32 i = start; i < end; i++) {
        T* ptr = array[i];

        if (ptr != nullptr) {
            delete ptr;
            array[i] = nullptr;
        }
    }
}

template<class T>
static inline void clearEntries(T** array, uint32 start, uint32 end) {
    for (uint32 i = start; i < end; i++) {
        T* ptr = array[i];

        if (ptr != nullptr) {
            ptr->clear();
        }
    }
}

template<class T>
DenseMappingVector<T>::Iterator::Iterator(const DenseMappingVector<T>& vector, uint32 index)
    : vector_(vector), index_(index) {

}

template<class T>
typename DenseMappingVector<T>::Iterator::reference DenseMappingVector<T>::Iterator::operator[](uint32 index) {
    Entry* ptr = vector_.array_[index];

    if (ptr == nullptr) {
        ptr = new Entry();
        vector_.array_[index] = ptr;
    }

    return *ptr;
}

template<class T>
typename DenseMappingVector<T>::Iterator::reference DenseMappingVector<T>::Iterator::operator*() {
    Entry* ptr = vector_.array_[index_];

    if (ptr == nullptr) {
        ptr = new Entry();
        vector_.array_[index_] = ptr;
    }

    return *ptr;
}

template<class T>
typename DenseMappingVector<T>::Iterator& DenseMappingVector<T>::Iterator::operator++() {
    ++index_;
    return *this;
}

template<class T>
typename DenseMappingVector<T>::Iterator& DenseMappingVector<T>::Iterator::operator++(int n) {
    index_++;
    return *this;
}

template<class T>
typename DenseMappingVector<T>::Iterator& DenseMappingVector<T>::Iterator::operator--() {
    --index_;
    return *this;
}

template<class T>
typename DenseMappingVector<T>::Iterator& DenseMappingVector<T>::Iterator::operator--(int n) {
    index_--;
    return *this;
}

template<class T>
bool DenseMappingVector<T>::Iterator::operator!=(const DenseMappingVector<T>::Iterator& rhs) const {
    return index_ != rhs.index_;
}

template<class T>
typename DenseMappingVector<T>::Iterator::difference_type DenseMappingVector<T>::Iterator::operator-(
        const DenseMappingVector<T>::Iterator& rhs) const {
    return (difference_type) index_ - (difference_type) rhs.index_;
}

template<class T>
DenseMappingVector<T>::ConstIterator::ConstIterator(const DenseMappingVector<T>& vector, uint32 index)
    : vector_(vector), index_(index) {

}

template<class T>
typename DenseMappingVector<T>::ConstIterator::reference DenseMappingVector<T>::ConstIterator::operator[](
        uint32 index) const {
    Entry* ptr = vector_.array_[index];
    return ptr != nullptr ? *ptr : vector_.emptyEntry_;
}

template<class T>
typename DenseMappingVector<T>::ConstIterator::reference DenseMappingVector<T>::ConstIterator::operator*() const {
    Entry* ptr = vector_.array_[index_];
    return ptr != nullptr ? *ptr : vector_.emptyEntry_;
}

template<class T>
typename DenseMappingVector<T>::ConstIterator& DenseMappingVector<T>::ConstIterator::operator++() {
    ++index_;
    return *this;
}

template<class T>
typename DenseMappingVector<T>::ConstIterator& DenseMappingVector<T>::ConstIterator::operator++(int n) {
    index_++;
    return *this;
}

template<class T>
typename DenseMappingVector<T>::ConstIterator& DenseMappingVector<T>::ConstIterator::operator--() {
    --index_;
    return *this;
}

template<class T>
typename DenseMappingVector<T>::ConstIterator& DenseMappingVector<T>::ConstIterator::operator--(int n) {
    index_--;
    return *this;
}

template<class T>
bool DenseMappingVector<T>::ConstIterator::operator!=(const ConstIterator& rhs) const {
    return index_ != rhs.index_;
}

template<class T>
typename DenseMappingVector<T>::ConstIterator::difference_type DenseMappingVector<T>::ConstIterator::operator-(
        const DenseMappingVector<T>::ConstIterator& rhs) const {
    return (difference_type) index_ - (difference_type) rhs.index_;
}

template<class T>
DenseMappingVector<T>::DenseMappingVector(uint32 numElements)
    : array_((Entry**) calloc(numElements, sizeof(Entry*))), numElements_(numElements),
      maxCapacity_(numElements) {

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
void DenseMappingVector<T>::clear() {
    clearEntries<Entry>(array_, 0, numElements_);
}

template<class T>
void DenseMappingVector<T>::swap(uint32 pos1, uint32 pos2) {
    if (pos1 != pos2) {
        Entry* ptr1 = array_[pos1];
        Entry* ptr2 = array_[pos2];
        array_[pos1] = ptr2;
        array_[pos2] = ptr1;
    }
}

template class DenseMappingVector<IndexedValue<float32>>;
template class DenseMappingVector<uint32>;
