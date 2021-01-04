/**
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "../data/types.h"


/**
 * Implements row-wise read-only access to the values that are stored in a pre-allocated C-contiguous array.
 *
 * @tparam T The type of the values
 */
template<class T>
class CContiguousView {

    private:

        uint32 numRows_;

        uint32 numCols_;

        const T* array_;

    public:

        /**
         * @param numRows   The number of rows in the view
         * @param numCols   The number of cols in the view
         * @param data      A pointer to a C-contiguous array of template type `T` that stores the values, the view
         *                  provides access to
         */
        CContiguousView(uint32 numRows, uint32 numCols, const T* array);

        typedef const T* const_iterator;

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
         * Returns the number of rows in the view.
         *
         * @return The number of rows
         */
        uint32 getNumRows() const;

        /**
         * Returns the number of columns in the view.
         *
         * @return The number of columns
         */
        uint32 getNumCols() const;

};
