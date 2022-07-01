/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/data/indexed_value.hpp"
#include <vector>


/**
 * A two-dimensional matrix that provides row-wise access to values that are stored in the list of lists (LIL) format.
 *
 * @tparam T The type of the values that are stored in the matrix
 */
template<typename T>
class SparseSetMatrix {

    public:

        /**
         * Provides access to a single row in the matrix.
         */
        class Row final {

            private:

                std::vector<IndexedValue<T>>& values_;

                uint32* indices_;

            public:

                /**
                 * @param values    A reference to the `std::vector` that stores the non-zero elements in the row
                 * @param indices   A pointer to an array of type `uint32` that stores for each element the
                 *                  corresponding index in `values`
                 */
                Row(std::vector<IndexedValue<T>>& values, uint32* indices);

                /**
                 * An iterator that provides access to the elements in the row and allows to modify them.
                 */
                typedef typename std::vector<IndexedValue<T>>::iterator iterator;

                /**
                 * An iterator that provides read-only access to the elements in the row.
                 */
                typedef typename std::vector<IndexedValue<T>>::const_iterator const_iterator;

                /**
                 * Returns an `iterator` to the beginning of the row.
                 *
                 * @return An `iterator` to the beginning
                 */
                iterator begin();

                /**
                 * Returns an `iterator` to the end of the row.
                 *
                 * @return An `iterator` to the end
                 */
                iterator end();

                /**
                 * Returns a `const_iterator` to the beginning of the row.
                 *
                 * @return A `const_iterator` to the beginning
                 */
                const_iterator cbegin() const;

                /**
                 * Returns a `const_iterator` to the end of the row.
                 *
                 * @return A `const_iterator` to the end
                 */
                const_iterator cend() const;

                /**
                 * Returns the number of non-zero elements in the row.
                 *
                 * @return The number of non-zero elements in the row
                 */
                uint32 getNumElements() const;

                /**
                 * Returns a pointer to the element that corresponds to a specific index.
                 *
                 * @param index The index of the element to be returned
                 * @return      A pointer to the element that corresponds to the given index or a null pointer, if no
                 *              such element is available
                 */
                const IndexedValue<T>* operator[](uint32 index) const;

                /**
                 * Returns a reference to the element that corresponds to a specific index. If no such element is
                 * available, it is inserted into the vector.
                 *
                 * @param index The index of the element to be returned
                 * @return      A reference to the element that corresponds to the given index
                 */
                IndexedValue<T>& emplace(uint32 index);

                /**
                 * Returns a reference to the element that corresponds to a specific index. If no such element is
                 * available, it is inserted into the vector using a specific default value.
                 *
                 * @param index         The index of the element to be returned
                 * @param defaultValue  The default value to be used
                 * @return              A reference to the element that corresponds to the given index
                 */
                IndexedValue<T>& emplace(uint32 index, const T& defaultValue);

                /**
                 * Removes the element that corresponds to a specific index, if available.
                 *
                 * @param index The index of the element to be removed
                 */
                void erase(uint32 index);

                /**
                 * Removes all elements from the vector.
                 */
                void clear();

        };

    private:

        uint32 numRows_;

        uint32 numCols_;

        std::vector<IndexedValue<T>>* values_;

        uint32* indices_;

    public:

        /**
         * @param numRows   The number of rows in the matrix
         * @param numCols   The number of columns in the matrix
         */
        SparseSetMatrix(uint32 numRows, uint32 numCols);

        virtual ~SparseSetMatrix();

        /**
         * An iterator that provides access to the elements at a row and allows to modify them.
         */
        typedef typename Row::iterator iterator;

        /**
         * An iterator that provides read-only access to the elements at a row.
         */
        typedef typename Row::const_iterator const_iterator;

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
         * Returns a specific row.
         *
         * @param row   The index of the row to be returned
         * @return      The row
         */
        Row getRow(uint32 row);

        /**
         * Returns a specific row.
         *
         * @param row   The index of the row to be returned
         * @return      The row
         */
        const Row getRow(uint32 row) const;

        /**
         * Returns the number of rows in the matrix.
         *
         * @return The number of rows
         */
        uint32 getNumRows() const;

        /**
         * Returns the number of columns in the matrix.
         *
         * @return The number of columns
         */
        uint32 getNumCols() const;

        /**
         * Sets the values of all elements to zero.
         */
        void clear();

};
