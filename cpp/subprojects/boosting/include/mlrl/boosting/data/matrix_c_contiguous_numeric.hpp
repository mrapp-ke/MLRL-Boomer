/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/matrix_c_contiguous.hpp"
#include "mlrl/common/indices/index_vector_complete.hpp"
#include "mlrl/common/indices/index_vector_partial.hpp"

namespace boosting {

    /**
     * A two-dimensional matrix that provides random access to a fixed number of values stored in a C-contiguous array.
     *
     * @tparam T The type of the values that are stored in the matrix
     */
    template<typename T>
    class NumericCContiguousMatrix final : public DenseMatrixDecorator<AllocatedCContiguousView<T>> {
        public:

            /**
             * @param numRows   The number of rows in the matrix
             * @param numCols   The number of columns in the matrix
             * @param init      True, if all elements in the matrix should be value-initialized, false otherwise
             */
            NumericCContiguousMatrix(uint32 numRows, uint32 numCols, bool init = false);

            /**
             * The type of the values that are stored by the matrix.
             */
            typedef T value_type;

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
            void addToRowFromSubset(uint32 row, typename View<T>::const_iterator begin,
                                    typename View<T>::const_iterator end,
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
            void addToRowFromSubset(uint32 row, typename View<T>::const_iterator begin,
                                    typename View<T>::const_iterator end,
                                    PartialIndexVector::const_iterator indicesBegin,
                                    PartialIndexVector::const_iterator indicesEnd);

            /**
             * Subtracts all values in another vector from certain elements, whose positions are given as a
             * `CompleteIndexVector`, at a specific row of this matrix.
             *
             * @param row           The row
             * @param begin         An iterator to the beginning of the vector
             * @param end           An iterator to the end of the vector
             * @param indicesBegin  An iterator to the beginning of the indices
             * @param indicesEnd    An iterator to the end of the indices
             */
            void removeFromRowFromSubset(uint32 row, typename View<T>::const_iterator begin,
                                         typename View<T>::const_iterator end,
                                         CompleteIndexVector::const_iterator indicesBegin,
                                         CompleteIndexVector::const_iterator indicesEnd);

            /**
             * Subtracts all values in another vector from certain elements, whose positions are given as a
             * `PartialIndexVector`, at a specific row of this matrix.
             *
             * @param row           The row
             * @param begin         An iterator to the beginning of the vector
             * @param end           An iterator to the end of the vector
             * @param indicesBegin  An iterator to the beginning of the indices
             * @param indicesEnd    An iterator to the end of the indices
             */
            void removeFromRowFromSubset(uint32 row, typename View<T>::const_iterator begin,
                                         typename View<T>::const_iterator end,
                                         PartialIndexVector::const_iterator indicesBegin,
                                         PartialIndexVector::const_iterator indicesEnd);
    };

}
