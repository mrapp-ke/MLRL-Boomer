/**
 * Provides classes that provide access to numerical data that is stored in matrices or vectors.
 *
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "../../common/cpp/data.h"
#include "../../common/cpp/indices.h"


namespace boosting {

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
             * Adds all numbers in another vector to certain elements, whose positions are given as a `FullIndexVector`,
             * at a specific row of this matrix.
             *
             * @param row           The row
             * @param begin         A `DenseVector<T>::const_iterator` to the beginning of the vector
             * @param end           A `DenseVector<T>::const_iterator` to the end of the vector
             * @param indicesBegin  A `FullIndexVector::const_iterator` to the beginning of the indices
             * @param indicesEnd    A `FullIndexVector::const_iterator` to the end of the indices
             */
            void addToRowFromSubset(uint32 row, typename DenseVector<T>::const_iterator begin,
                                    typename DenseVector<T>::const_iterator end,
                                    FullIndexVector::const_iterator indicesBegin,
                                    FullIndexVector::const_iterator indicesEnd);

            /**
             * Adds all numbers in another vector to certain elements, whose positions are given as a
             * `PartialIndexVector`, at a specific row of this matrix.
             *
             * @param row           The row
             * @param begin         A `DenseVector<T>::const_iterator` to the beginning of the vector
             * @param end           A `DenseVector<T>::const_iterator` to the end of the vector
             * @param indicesBegin  A `PartialIndexVector::const_iterator` to the beginning of the indices
             * @param indicesEnd    A `PartialIndexVector::const_iterator` to the end of the indices
             */
            void addToRowFromSubset(uint32 row, typename DenseVector<T>::const_iterator begin,
                                    typename DenseVector<T>::const_iterator end,
                                    PartialIndexVector::const_iterator indicesBegin,
                                    PartialIndexVector::const_iterator indicesEnd);

    };

}
