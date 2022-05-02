#include "boosting/data/matrix_lil_numeric.hpp"
#include <iostream>  // TODO Remove


namespace boosting {

    template<typename T>
    NumericLilMatrix<T>::NumericLilMatrix(uint32 numRows, uint32 numCols)
        : LilMatrix<T>(numRows, numCols) {

    }

    template<typename T>
    void NumericLilMatrix<T>::addToRowFromSubset(uint32 row, typename VectorConstView<T>::const_iterator begin,
                                                 typename VectorConstView<T>::const_iterator end,
                                                 CompleteIndexVector::const_iterator indicesBegin,
                                                 CompleteIndexVector::const_iterator indicesEnd) {
        // TODO Implement
        std::cout << "NumericLilMatrix::addToRowFromSubset(CompleteIndexVector)\n";
        std::exit(-1);
    }

    template<typename T>
    void NumericLilMatrix<T>::addToRowFromSubset(uint32 row, typename VectorConstView<T>::const_iterator begin,
                                                 typename VectorConstView<T>::const_iterator end,
                                                 PartialIndexVector::const_iterator indicesBegin,
                                                 PartialIndexVector::const_iterator indicesEnd) {
        // TODO Implement
        std::cout << "NumericLilMatrix::addToRowFromSubset(PartialIndexVector)\n";
        std::exit(-1);
    }

    template class NumericLilMatrix<uint8>;
    template class NumericLilMatrix<uint32>;
    template class NumericLilMatrix<float32>;
    template class NumericLilMatrix<float64>;

}
