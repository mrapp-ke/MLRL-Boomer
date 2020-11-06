/**
 * Provides classes that provide access to numerical data that is stored in matrices or vectors.
 *
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "data.h"
#include "indices.h"


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

        /**
         * Adds all numbers in another vector to this vector.
         *
         * @param begin A `DenseVector<T>::const_iterator` to the beginning of the other vector
         * @param end   A `DenseVector<T>::const_iterator` to the end of the other vector
         */
        void add(typename DenseVector<T>::const_iterator begin, typename DenseVector<T>::const_iterator end);

        /**
         * Adds all numbers in another vector to this vector. The numbers to be added are multiplied by a specific
         * weight.
         *
         * @param begin     A `DenseVector<T>::const_iterator` to the beginning of the other vector
         * @param end       A `DenseVector<T>::const_iterator` to the end of the other vector
         * @param weight    The weight, the numbers should be multiplied by
         */
        void add(typename DenseVector<T>::const_iterator begin, typename DenseVector<T>::const_iterator end, T weight);

        /**
         * Adds certain numbers in another vector, whose positions are given as a `FullIndexVector`, to this vector. The
         * numbers to be added are multiplied by a specific weight.
         *
         * @param begin     A `DenseVector<T>::const_iterator` to the beginning of the other vector
         * @param end       A `DenseVector<T>::const_iterator` to the end of the other vector
         * @param indices   A reference to a `FullIndexVector' that provides access to the indices
         * @param weight    The weight, the numbers should be multiplied by
         */
        void addToSubset(typename DenseVector<T>::const_iterator begin, typename DenseVector<T>::const_iterator end,
                         const FullIndexVector& indices, T weight);

        /**
         * Adds certain numbers in another vector, whose positions are given as a `PartialIndexVector`, to this vector.
         * The numbers to be added are multiplied by a specific weight.
         *
         * @param begin     A `DenseVector<T>::const_iterator` to the beginning of the other vector
         * @param end       A `DenseVector<T>::const_iterator` to the end of the other vector
         * @param indices   A reference to a `PartialIndexVector' that provides access to the indices
         * @param weight    The weight, the numbers should be multiplied by
         */
        void addToSubset(typename DenseVector<T>::const_iterator begin, typename DenseVector<T>::const_iterator end,
                         const PartialIndexVector& indices, T weight);

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
