/**
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "types.h"


/**
 * A two-dimensional matrix that provides random access to a fixed number of elements stored in a C-contiguous array.
 *
 * @tparam T The type of the data that is stored in the matrix
 */
template<class T>
class DenseMatrix {

    protected:

        T* array_;

        uint32 numRows_;

        uint32 numCols_;

    public:

        /**
         * @param numRows   The number of rows in the matrix
         * @param numCols   The number of columns in the matrix
         */
        DenseMatrix(uint32 numRows, uint32 numCols);

        /**
         * @param numRows   The number of rows in the matrix
         * @param numCols   The number of columns in the matrix
         * @param init      True, if all elements in the matrix should be value-initialized, false otherwise
         */
        DenseMatrix(uint32 numRows, uint32 numCols, bool init);

        ~DenseMatrix();

        typedef T* iterator;

        typedef const T* const_iterator;

        /**
         * Returns an `iterator` to the beginning of a specific row.
         *
         * @param row   The row
         * @return      An `iterator` to the beginning of the given row
         */
        iterator row_begin(uint32 row);

        /**
         * Returns an `iterator` to the end of a specific row.
         *
         * @param row   The row
         * @return      An `iterator` to the end of the given row
         */
        iterator row_end(uint32 row);

        /**
         * Returns a `const_iterator` to the beginning of a specific row.
         *
         * @param row   The row
         * @return      A `const_iterator` to the beginning of the given row
         */
        const_iterator row_cbegin(uint32 row) const;

        /**
         * Returns a `const_iterator` to the end of a specific row.
         *
         * @param row   The row
         * @return      A `const_iterator` to the end of the given row
         */
        const_iterator row_cend(uint32 row) const;

        /**
         * Returns the number of rows in the matrix.
         *
         * @return The number of rows in the matrix
         */
        uint32 getNumRows() const;

        /**
         * Returns the number of columns in the matrix.
         *
         * @return The number of columns in the matrix
         */
        uint32 getNumCols() const;

};
