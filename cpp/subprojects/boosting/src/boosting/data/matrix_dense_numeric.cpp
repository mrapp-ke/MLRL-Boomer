#include "boosting/data/matrix_dense_numeric.hpp"


namespace boosting {

    template<class T>
    DenseNumericMatrix<T>::DenseNumericMatrix(uint32 numRows, uint32 numCols)
        : DenseMatrix<T>(numRows, numCols) {

    }

    template<class T>
    DenseNumericMatrix<T>::DenseNumericMatrix(uint32 numRows, uint32 numCols, bool init)
        : DenseMatrix<T>(numRows, numCols, init) {

    }

    template<class T>
    void DenseNumericMatrix<T>::addToRowFromSubset(uint32 row, typename DenseVector<T>::const_iterator begin,
                                                   typename DenseVector<T>::const_iterator end,
                                                   FullIndexVector::const_iterator indicesBegin,
                                                   FullIndexVector::const_iterator indicesEnd) {
        typename DenseNumericMatrix<T>::iterator iterator = this->row_begin(row);
        uint32 numCols = this->getNumCols();

        for (uint32 i = 0; i < numCols; i++) {
            iterator[i] += begin[i];
        }
    }

    template<class T>
    void DenseNumericMatrix<T>::addToRowFromSubset(uint32 row, typename DenseVector<T>::const_iterator begin,
                                                   typename DenseVector<T>::const_iterator end,
                                                   PartialIndexVector::const_iterator indicesBegin,
                                                   PartialIndexVector::const_iterator indicesEnd) {
        typename DenseNumericMatrix<T>::iterator iterator = this->row_begin(row);
        uint32 numCols = indicesEnd - indicesBegin;

        for (uint32 i = 0; i < numCols; i++) {
            uint32 index = indicesBegin[i];
            iterator[index] += begin[i];
        }
    }

    template class DenseNumericMatrix<uint8>;
    template class DenseNumericMatrix<uint32>;
    template class DenseNumericMatrix<float32>;
    template class DenseNumericMatrix<float64>;

}
