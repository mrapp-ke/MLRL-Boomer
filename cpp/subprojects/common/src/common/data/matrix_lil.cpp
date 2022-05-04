#include "common/data/matrix_lil.hpp"
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
LilMatrix<T>::Row::Row(std::vector<IndexedValue<T>>& values, uint32* indices)
    : values_(values), indices_(indices) {

}

template<typename T>
typename LilMatrix<T>::Row::iterator LilMatrix<T>::Row::begin() {
    return values_.begin();
}

template<typename T>
typename LilMatrix<T>::Row::iterator LilMatrix<T>::Row::end() {
    return values_.end();
}

template<typename T>
typename LilMatrix<T>::Row::const_iterator LilMatrix<T>::Row::cbegin() const {
    return values_.cbegin();
}

template<typename T>
typename LilMatrix<T>::Row::const_iterator LilMatrix<T>::Row::cend() const {
    return values_.cend();
}

template<typename T>
uint32 LilMatrix<T>::Row::getNumElements() const {
    return (uint32) values_.size();
}

template<typename T>
const IndexedValue<T>* LilMatrix<T>::Row::operator[](uint32 index) const {
    uint32 i = indices_[index];
    return i == MAX_INDEX ? nullptr : &values_[i];
}

template<typename T>
IndexedValue<T>& LilMatrix<T>::Row::emplace(uint32 index) {
    uint32 i = indices_[index];

    if (i == MAX_INDEX) {
        indices_[index] = (uint32) values_.size();
        values_.emplace_back(index);
        return values_.back();
    }

    return values_[i];
}

template<typename T>
IndexedValue<T>& LilMatrix<T>::Row::emplace(uint32 index, const T& defaultValue) {
    uint32 i = indices_[index];

    if (i == MAX_INDEX) {
        indices_[index] = (uint32) values_.size();
        values_.emplace_back(index, defaultValue);
        return values_.back();
    }

    return values_[i];
}

template<typename T>
void LilMatrix<T>::Row::erase(uint32 index) {
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
void LilMatrix<T>::Row::clear() {
    clearRow(values_, indices_);
}

template<typename T>
LilMatrix<T>::LilMatrix(uint32 numRows, uint32 numCols)
    : numRows_(numRows), numCols_(numCols), values_(new std::vector<IndexedValue<T>>[numRows]),
      indices_(new uint32[numRows * numCols]) {
    setArrayToValue(indices_, numRows * numCols, MAX_INDEX);
}

template<typename T>
LilMatrix<T>::~LilMatrix() {
    delete[] values_;
    delete[] indices_;
}

template<typename T>
typename LilMatrix<T>::iterator LilMatrix<T>::row_begin(uint32 row) {
    return values_[row].begin();
}

template<typename T>
typename LilMatrix<T>::iterator LilMatrix<T>::row_end(uint32 row) {
    return values_[row].end();
}

template<typename T>
typename LilMatrix<T>::const_iterator LilMatrix<T>::row_cbegin(uint32 row) const {
    return values_[row].cbegin();
}

template<typename T>
typename LilMatrix<T>::const_iterator LilMatrix<T>::row_cend(uint32 row) const {
    return values_[row].cend();
}

template<typename T>
typename LilMatrix<T>::Row LilMatrix<T>::getRow(uint32 row) {
    return LilMatrix<T>::Row(values_[row], &indices_[row * numCols_]);
}

template<typename T>
const typename LilMatrix<T>::Row LilMatrix<T>::getRow(uint32 row) const {
    return LilMatrix<T>::Row(values_[row], &indices_[row * numCols_]);
}

template<typename T>
uint32 LilMatrix<T>::getNumRows() const {
    return numRows_;
}

template<typename T>
uint32 LilMatrix<T>::getNumCols() const {
    return numCols_;
}

template<typename T>
void LilMatrix<T>::clear() {
    for (uint32 i = 0; i < numRows_; i++) {
        clearRow(values_[i], &indices_[i * numCols_]);
    }
}

template class LilMatrix<uint8>;
template class LilMatrix<uint32>;
template class LilMatrix<float32>;
template class LilMatrix<float64>;
template class LilMatrix<Tuple<uint8>>;
template class LilMatrix<Tuple<uint32>>;
template class LilMatrix<Tuple<float32>>;
template class LilMatrix<Tuple<float64>>;
template class LilMatrix<Triple<uint8>>;
template class LilMatrix<Triple<uint32>>;
template class LilMatrix<Triple<float32>>;
template class LilMatrix<Triple<float64>>;
