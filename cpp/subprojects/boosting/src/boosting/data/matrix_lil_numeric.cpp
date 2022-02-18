#include "boosting/data/matrix_lil_numeric.hpp"


namespace boosting {

    template<typename T, typename IndexIterator>
    static inline void addInternally(typename NumericLilMatrix<T>::Row& row,
                                     typename VectorConstView<T>::const_iterator iterator,
                                     typename VectorConstView<T>::const_iterator end, IndexIterator indexIterator,
                                     IndexIterator indicesEnd) {
        uint32 numElements = indicesEnd - indexIterator;

        if (numElements > 0) {
            typename NumericLilMatrix<T>::Row::iterator previous = row.begin();
            typename NumericLilMatrix<T>::Row::iterator last = row.end();
            typename NumericLilMatrix<T>::Row::iterator current = addFirst<T>(row, previous, last, indexIterator[0],
                                                                              iterator[0]);
            uint32 i = 1;

            while (current != last) {
                if (i < numElements) {
                    add<T>(row, previous, current, last, indexIterator[i], iterator[i]);
                    i++;
                } else {
                    return;
                }
            }

            for (; i < numElements; i++) {
                previous = row.emplace_after(previous, indexIterator[i], iterator[i]);
            }
        }
    }

    template<typename T>
    NumericLilMatrix<T>::NumericLilMatrix(uint32 numRows)
        : LilMatrix<T>(numRows) {

    }

    template<typename T>
    void NumericLilMatrix<T>::addToRowFromSubset(uint32 row, typename VectorConstView<T>::const_iterator begin,
                                                 typename VectorConstView<T>::const_iterator end,
                                                 CompleteIndexVector::const_iterator indicesBegin,
                                                 CompleteIndexVector::const_iterator indicesEnd) {
        addInternally<T, CompleteIndexVector::const_iterator>(this->getRow(row), begin, end, indicesBegin, indicesEnd);
    }

    template<typename T>
    void NumericLilMatrix<T>::addToRowFromSubset(uint32 row, typename VectorConstView<T>::const_iterator begin,
                                                 typename VectorConstView<T>::const_iterator end,
                                                 PartialIndexVector::const_iterator indicesBegin,
                                                 PartialIndexVector::const_iterator indicesEnd) {
        addInternally<T, PartialIndexVector::const_iterator>(this->getRow(row), begin, end, indicesBegin, indicesEnd);
    }

    template class NumericLilMatrix<uint8>;
    template class NumericLilMatrix<uint32>;
    template class NumericLilMatrix<float32>;
    template class NumericLilMatrix<float64>;

}
