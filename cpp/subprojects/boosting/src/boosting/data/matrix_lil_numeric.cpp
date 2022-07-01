#include "boosting/data/matrix_lil_numeric.hpp"


namespace boosting {

    template<typename T, typename IndexIterator>
    static inline void addToRowFromSubsetInternally(typename NumericLilMatrix<T>::Row row,
                                                    typename VectorConstView<T>::const_iterator iterator,
                                                    IndexIterator indexIterator, uint32 numElements) {
        for (uint32 i = 0; i < numElements; i++) {
            uint32 index = indexIterator[i];
            IndexedValue<T>& entry = row.emplace(index, 0);
            entry.value += iterator[i];
        }
    }

    template<typename T>
    NumericLilMatrix<T>::NumericLilMatrix(uint32 numRows, uint32 numCols)
        : SparseSetMatrix<T>(numRows, numCols) {

    }

    template<typename T>
    void NumericLilMatrix<T>::addToRowFromSubset(uint32 row, typename VectorConstView<T>::const_iterator begin,
                                                 typename VectorConstView<T>::const_iterator end,
                                                 CompleteIndexVector::const_iterator indicesBegin,
                                                 CompleteIndexVector::const_iterator indicesEnd) {
        addToRowFromSubsetInternally<T, CompleteIndexVector::const_iterator>(this->getRow(row), begin, indicesBegin,
                                                                             this->getNumCols());
    }

    template<typename T>
    void NumericLilMatrix<T>::addToRowFromSubset(uint32 row, typename VectorConstView<T>::const_iterator begin,
                                                 typename VectorConstView<T>::const_iterator end,
                                                 PartialIndexVector::const_iterator indicesBegin,
                                                 PartialIndexVector::const_iterator indicesEnd) {
        uint32 numElements = indicesEnd - indicesBegin;
        addToRowFromSubsetInternally<T, PartialIndexVector::const_iterator>(this->getRow(row), begin, indicesBegin,
                                                                            numElements);
    }

    template class NumericLilMatrix<uint8>;
    template class NumericLilMatrix<uint32>;
    template class NumericLilMatrix<float32>;
    template class NumericLilMatrix<float64>;

}
