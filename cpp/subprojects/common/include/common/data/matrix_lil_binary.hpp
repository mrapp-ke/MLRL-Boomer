/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/data/types.hpp"
#include <vector>


/**
 * A two-dimensional matrix that provides row-wise access to binary values that are stored in the list of lists (LIL)
 * format.
 */
class BinaryLilMatrix final {

    private:

        uint32 numRows_;

        std::vector<uint32>* array_;

    public:

        /**
         * @param numRows The number of rows in the matrix
         */
        BinaryLilMatrix(uint32 numRows);

        ~BinaryLilMatrix();

        /**
         * Provides access to a row and allows to modify its elements.
         */
        typedef std::vector<uint32> row;

        /**
         * Provides read-only access to a row.
         */
        typedef const std::vector<uint32> const_row;

        /**
         * An iterator that provides access to the elements in the matrix and allows to modify them.
         */
        typedef std::vector<uint32>::iterator iterator;

        /**
         * An iterator that provides read-only access to the elements in the matrix.
         */
        typedef std::vector<uint32>::const_iterator const_iterator;

        /**
         * Returns an `iterator` to the beginning of a specific row.
         *
         * @param row   The row
         * @return      An `iterator` to the beginning
         */
        iterator row_begin(uint32 row);

        /**
         * Returns an `iterator` to the end of a specific row.
         *
         * @param row   The row
         * @return      An `iterator` to the end
         */
        iterator row_end(uint32 row);

        /**
         * Returns a `const_iterator` to the beginning of a specific row.
         *
         * @param row   The row
         * @return      A `const_iterator` to the beginning
         */
        const_iterator row_cbegin(uint32 row) const;

        /**
         * Returns a `const_iterator` to the end of a specific row.
         *
         * @param row   The row
         * @return      A `const_iterator` to the end
         */
        const_iterator row_cend(uint32 row) const;

        /**
         * Provides access to a specific row and allows to modify its elements.
         *
         * @param row   The index of the row
         * @return      A `row`
         */
        row& getRow(uint32 row);

        /**
         * Provides read-only access to a specific row.
         *
         * @param row   The index of the row
         * @return      A `const_row`
         */
        const_row& getRow(uint32 row) const;

        /**
         * Returns the number of rows in the matrix.
         *
         * @return The number of rows
         */
        uint32 getNumRows() const;

        /**
         * Sets the values of all elements to zero.
         */
        void clear();

};
