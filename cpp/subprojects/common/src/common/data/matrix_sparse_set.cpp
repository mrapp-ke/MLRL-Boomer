#include "common/data/matrix_sparse_set.hpp"
#include "common/data/arrays.hpp"
#include "common/data/tuple.hpp"
#include "common/data/triple.hpp"
#include <limits>


static const uint32 MAX_INDEX = std::numeric_limits<uint32>::max();

template<typename T>
static inline void clearRow(std::vector<IndexedValue<T>>& values, uint32* indices) {
    while (!values.empty()) {
        const IndexedValue<T>& lastEntry = values.back();
        indices[lastEntry.index] = MAX_INDEX;
        values.pop_back();
    }
}

template<typename T>
SparseSetMatrix<T>::Row::Row(std::vector<IndexedValue<T>>& values, uint32* indices)
    : values_(values), indices_(indices) {

}

template<typename T>
typename SparseSetMatrix<T>::Row::iterator SparseSetMatrix<T>::Row::begin() {
    return values_.begin();
}

template<typename T>
typename SparseSetMatrix<T>::Row::iterator SparseSetMatrix<T>::Row::end() {
    return values_.end();
}

template<typename T>
typename SparseSetMatrix<T>::Row::const_iterator SparseSetMatrix<T>::Row::cbegin() const {
    return values_.cbegin();
}

template<typename T>
typename SparseSetMatrix<T>::Row::const_iterator SparseSetMatrix<T>::Row::cend() const {
    return values_.cend();
}

template<typename T>
uint32 SparseSetMatrix<T>::Row::getNumElements() const {
    return (uint32) values_.size();
}

template<typename T>
const IndexedValue<T>* SparseSetMatrix<T>::Row::operator[](uint32 index) const {
    uint32 i = indices_[index];
    return i == MAX_INDEX ? nullptr : &values_[i];
}

template<typename T>
IndexedValue<T>& SparseSetMatrix<T>::Row::emplace(uint32 index) {
    uint32 i = indices_[index];

    if (i == MAX_INDEX) {
        indices_[index] = (uint32) values_.size();
        values_.emplace_back(index);
        return values_.back();
    }

    return values_[i];
}

template<typename T>
IndexedValue<T>& SparseSetMatrix<T>::Row::emplace(uint32 index, const T& defaultValue) {
    uint32 i = indices_[index];

    if (i == MAX_INDEX) {
        indices_[index] = (uint32) values_.size();
        values_.emplace_back(index, defaultValue);
        return values_.back();
    }

    return values_[i];
}

template<typename T>
void SparseSetMatrix<T>::Row::erase(uint32 index) {
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
void SparseSetMatrix<T>::Row::clear() {
    clearRow(values_, indices_);
}

template<typename T>
SparseSetMatrix<T>::SparseSetMatrix(uint32 numRows, uint32 numCols)
    : numRows_(numRows), numCols_(numCols), values_(new std::vector<IndexedValue<T>>[numRows]),
      indices_(new uint32[numRows * numCols]) {
    setArrayToValue(indices_, numRows * numCols, MAX_INDEX);
}

template<typename T>
SparseSetMatrix<T>::~SparseSetMatrix() {
    delete[] values_;
    delete[] indices_;
}

template<typename T>
typename SparseSetMatrix<T>::iterator SparseSetMatrix<T>::row_begin(uint32 row) {
    return values_[row].begin();
}

template<typename T>
typename SparseSetMatrix<T>::iterator SparseSetMatrix<T>::row_end(uint32 row) {
    return values_[row].end();
}

template<typename T>
typename SparseSetMatrix<T>::const_iterator SparseSetMatrix<T>::row_cbegin(uint32 row) const {
    return values_[row].cbegin();
}

template<typename T>
typename SparseSetMatrix<T>::const_iterator SparseSetMatrix<T>::row_cend(uint32 row) const {
    return values_[row].cend();
}

template<typename T>
typename SparseSetMatrix<T>::Row SparseSetMatrix<T>::getRow(uint32 row) {
    return SparseSetMatrix<T>::Row(values_[row], &indices_[row * numCols_]);
}

template<typename T>
const typename SparseSetMatrix<T>::Row SparseSetMatrix<T>::getRow(uint32 row) const {
    return SparseSetMatrix<T>::Row(values_[row], &indices_[row * numCols_]);
}

template<typename T>
uint32 SparseSetMatrix<T>::getNumRows() const {
    return numRows_;
}

template<typename T>
uint32 SparseSetMatrix<T>::getNumCols() const {
    return numCols_;
}

template<typename T>
void SparseSetMatrix<T>::clear() {
    for (uint32 i = 0; i < numRows_; i++) {
        clearRow(values_[i], &indices_[i * numCols_]);
    }
}

template class SparseSetMatrix<uint8>;
template class SparseSetMatrix<uint32>;
template class SparseSetMatrix<float32>;
template class SparseSetMatrix<float64>;
template class SparseSetMatrix<Tuple<uint8>>;
template class SparseSetMatrix<Tuple<uint32>>;
template class SparseSetMatrix<Tuple<float32>>;
template class SparseSetMatrix<Tuple<float64>>;
template class SparseSetMatrix<Triple<uint8>>;
template class SparseSetMatrix<Triple<uint32>>;
template class SparseSetMatrix<Triple<float32>>;
template class SparseSetMatrix<Triple<float64>>;
