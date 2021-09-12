#include "boosting/data/matrix_lil_numeric.hpp"


namespace boosting {

    template<typename T>
    NumericLilMatrix<T>::NumericLilMatrix(uint32 numRows)
        : LilMatrix<T>(numRows) {

    }

    template<typename T>
    void NumericLilMatrix<T>::addToRowFromSubset(uint32 row, typename DenseVector<T>::const_iterator begin,
                                                 typename DenseVector<T>::const_iterator end,
                                                 CompleteIndexVector::const_iterator indicesBegin,
                                                 CompleteIndexVector::const_iterator indicesEnd) {
        // TODO Implement
    }

    template<typename T>
    void NumericLilMatrix<T>::addToRowFromSubset(uint32 row, typename DenseVector<T>::const_iterator begin,
                                                 typename DenseVector<T>::const_iterator end,
                                                 PartialIndexVector::const_iterator indicesBegin,
                                                 PartialIndexVector::const_iterator indicesEnd) {
        // TODO Implement
    }

    template class NumericLilMatrix<uint8>;
    template class NumericLilMatrix<uint32>;
    template class NumericLilMatrix<float32>;
    template class NumericLilMatrix<float64>;

}
