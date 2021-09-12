/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/data/matrix_lil.hpp"
#include "common/indices/index_vector_complete.hpp"
#include "common/indices/index_vector_partial.hpp"


namespace boosting {

    /**
     * A two-dimensional matrix that provides row-wise access to values that are stored in the list of lists (LIL)
     * format.
     *
     * @tparam T The type of the values that are stored in the matrix
     */
    template<typename T>
    class NumericLilMatrix final : public LilMatrix<T> {

        public:

            /**
             * @param numRows The number of rows in the matrix
             */
            NumericLilMatrix(uint32 numRows);

            /**
             * Adds all values in another vector to certain elements, whose positions are given as a
             * `CompleteIndexVector`, at a specific row of this matrix.
             *
             * @param row           The row
             * @param begin         An iterator to the beginning of the vector
             * @param end           An iterator to the end of the vector
             * @param indicesBegin  An iterator to the beginning of the indices
             * @param indicesEnd    An iterator to the end of the indices
             */
            void addToRowFromSubset(uint32 row, typename DenseVector<T>::const_iterator begin,
                                    typename DenseVector<T>::const_iterator end,
                                    CompleteIndexVector::const_iterator indicesBegin,
                                    CompleteIndexVector::const_iterator indicesEnd);

            /**
             * Adds all values in another vector to certain elements, whose positions are given as a
             * `PartialIndexVector`, at a specific row of this matrix.
             *
             * @param row           The row
             * @param begin         An iterator to the beginning of the vector
             * @param end           An iterator to the end of the vector
             * @param indicesBegin  An iterator to the beginning of the indices
             * @param indicesEnd    An iterator to the end of the indices
             */
            void addToRowFromSubset(uint32 row, typename DenseVector<T>::const_iterator begin,
                                    typename DenseVector<T>::const_iterator end,
                                    PartialIndexVector::const_iterator indicesBegin,
                                    PartialIndexVector::const_iterator indicesEnd);

    };

}
