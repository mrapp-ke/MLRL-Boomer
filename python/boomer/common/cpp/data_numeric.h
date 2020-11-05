/**
 * Provides classes that provide access to numerical data that is stored in matrices or vectors.
 *
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "data.h"


/**
 * An one-dimensional vector that provides random access to a fixed number of numbers stored in a C-contiguous array.
 *
 * @tparam T The type of the numbers that are stored in the vector
 */
template<class T>
class DenseNumericVector : public DenseVector<T> {

    public:

        /**
         * @param numElements The number of elements in the vector
         */
        DenseNumericVector(uint32 numElements);

        /**
         * @param numElements   The number of elements in the vector
         * @param init          True, if all elements in the vector should be value-initialized, false otherwise
         */
        DenseNumericVector(uint32 numElements, bool init);

        /**
         * Sets the values of all elements in the vector to zero.
         */
        void setAllToZero();

};

typedef DenseNumericVector<float64> DenseFloat64Vector;

/**
 * A two-dimensional matrix that provides random access to a fixed number of numbers stored in a C-contiguous array.
 *
 * @tparam T The type of the numbers that are stored in the matrix
 */
template<class T>
class DenseNumericMatrix : public DenseMatrix<T> {

    public:

        /**
         * @param numRows   The number of rows in the matrix
         * @param numCols   The number of columns in the matrix
         */
        DenseNumericMatrix(uint32 numRows, uint32 numCols);

        /**
         * @param numRows   The number of rows in the matrix
         * @param numCols   The number of columns in the matrix
         * @param init      True, if all elements in the matrix should be value-initialized, false otherwise
         */
        DenseNumericMatrix(uint32 numRows, uint32 numCols, bool init);

        /**
         * Sets the values of all elements in the matrix to zero.
         */
        void setAllToZero();

};

typedef DenseNumericMatrix<float64> DenseFloat64Matrix;
