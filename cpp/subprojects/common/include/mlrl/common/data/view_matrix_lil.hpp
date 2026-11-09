/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/view_matrix.hpp"
#include "mlrl/common/data/view_vector.hpp"

#include <utility>

/**
 * A two-dimensional view that provides row-wise access to values stored in a sparse matrix in the list of lists (LIL)
 * format.
 *
 * @tparam T The type of the values, the view provides access to
 */
template<typename T>
class MLRLCOMMON_API ListOfLists : public Matrix {
    public:

        /**
         * A one-dimensional view that provides access to the values stored in a single row of a matrix in the list of
         * lists (LIL) format.
         */
        class Row final : public Vector<T> {
            public:

                Row() : Vector<T>(nullptr, 0, 0) {}

                /**
                 * @param other A const reference to an object of type `Row` that should be copied
                 */
                Row(const Row& other) : Vector<T>(other) {}

                /**
                 * @param other A reference to an object of type `Row` that should be moved
                 */
                Row(Row&& other) : Vector<T>(std::move(other)) {}
        };

        /**
         * A pointer to an array that stores vectors corresponding to each row in the matrix.
         */
        Row* array;

        /**
         * @param array     A pointer to an array of type `row`, shape `(numRows)`, that stores vectors corresponding
         *                  to each row in the matrix
         * @param numRows   The number of rows in the view
         * @param numCols   The number of columns in the view
         */
        ListOfLists(Row* array, uint32 numRows, uint32 numCols) : Matrix(numRows, numCols), array(array) {}

        /**
         * @param other A const reference to an object of type `ListOfLists` that should be copied
         */
        ListOfLists(const ListOfLists& other) : Matrix(other), array(other.array) {}

        /**
         * @param other A reference to an object of type `ListOfLists` that should be moved
         */
        ListOfLists(ListOfLists&& other) : Matrix(std::move(other)), array(other.array) {}

        virtual ~ListOfLists() override {}

        /**
         * The type of the values, the view provides access to.
         */
        using value_type = Row::value_type;

        /**
         * An iterator that provides read-only access to the values in the view.
         */
        using value_const_iterator = Row::const_iterator;

        /**
         * An iterator that provides access to the values in the view and allows to modify them.
         */
        using value_iterator = Row::iterator;

        /**
         * Provides read-only access to an individual row in the view.
         */
        using const_row = const Row&;

        /**
         * Provides access to an individual row in the view and allows to modify it.
         */
        using row = Row&;

        /**
         * Returns a view that provides read-only access to a specific row in the view.
         *
         * @param row   The index of the row
         * @return      A `const_row`
         */
        const_row operator[](uint32 row) const {
            return array[row];
        }

        /**
         * Returns a view that provides access to a specific row in the view and allows to modify it.
         *
         * @param row   The index of the row
         * @return      A `row`
         */
        row operator[](uint32 row) {
            return array[row];
        }

        /**
         * Returns a `value_const_iterator` to the beginning of the values in a specific row of the matrix.
         *
         * @param row   The index of the row
         * @return      A `value_const_iterator` to the beginning of the values
         */
        value_const_iterator values_cbegin(uint32 row) const {
            return array[row].cbegin();
        }

        /**
         * Returns a `value_const_iterator` to the end of the values in a specific row of the matrix.
         *
         * @param row   The index of the row
         * @return      A `value_const_iterator` to the end of the values
         */
        value_const_iterator values_cend(uint32 row) const {
            return array[row].cend();
        }

        /**
         * Returns a `value_iterator` to the beginning of the values in a specific row of the matrix.
         *
         * @param row   The index of the row
         * @return      A `value_iterator` to the beginning of the values
         */
        value_iterator values_begin(uint32 row) {
            return array[row].begin();
        }

        /**
         * Returns a `value_iterator` to the end of the values in a specific row of the matrix.
         *
         * @param row   The index of the row
         * @return      A `value_iterator` to the end of the values
         */
        value_iterator values_end(uint32 row) {
            return array[row].end();
        }

        /**
         * Sets all values stored in the matrix to zero.
         */
        void clear() {
            for (uint32 i = 0; i < numRows; i++) {
                row row = (*this)[i];
                row.numElements = 0;
                row.numElementsWithPadding = 0;
            }
        }

        /**
         * Releases the ownership of the array that stores vectors corresponding to each row in the matrix. As a result,
         * the behavior of this view becomes undefined and it should not be used anymore. The caller is responsible for
         * freeing the memory that is occupied by the array.
         *
         * @return A pointer to the array that stores vectors corresponding to each row in the matrix
         */
        ListOfLists<T>::Row* release() {
            Row* ptr = array;
            array = nullptr;
            return ptr;
        }
};

/**
 * Allocates the memory for a two-dimensional view that provides row-wise access to values stored in a matrix in the
 * list of lists (LIL) format.
 *
 * @tparam Matrix           The type of the view
 * @tparam MemoryAllocator  The type of the memory allocator to be used
 */
template<typename Matrix, typename MemoryAllocator = DefaultMemoryAllocator>
class MLRLCOMMON_API ListOfListsAllocator : public Matrix {
    private:

        uint32* maxCapacities_;

    public:

        /**
         * A one-dimensional view that provides access to the values stored in a single row of a matrix in the list of
         * lists (LIL) format and allows to add or remove elements to this row.
         */
        class Row final : public Vector<typename Matrix::value_type> {
            private:

                Matrix::Row& row_;

                uint32* maxCapacity_;

            public:

                /**
                 * @param row           A reference to an object of type `Matrix::Row` that provides access to the row
                 * @param maxCapacity   A pointer to the value that specifies the maximum capacity of the row
                 */
                Row(Matrix::Row& row, uint32* maxCapacity)
                    : Vector<typename Matrix::value_type>(row.array, row.numElements,
                                                          row.numElementsWithPadding - row.numElements),
                      row_(row), maxCapacity_(maxCapacity) {}

                /**
                 * @param other A const reference to an object of type `Row` that should be copied
                 */
                Row(const Row& other)
                    : Vector<typename Matrix::value_type>(other), row_(other.row_), maxCapacity_(other.maxCapacity_) {}

                /**
                 * @param other A reference to an object of type `Row` that should be moved
                 */
                Row(Row&& other)
                    : Vector<typename Matrix::value_type>(std::move(other)), row_(other.row_),
                      maxCapacity_(other.maxCapacity_) {}

                /**
                 * Reserves memory for a certain number of elements, including padding, if necessary.
                 *
                 * @param numElements   The number of elements for which memory should be reserved
                 * @return              The number of elements for which memory is available, including padding
                 */
                uint32 reserve(uint32 numElements) {
                    const uint32 numElementsWithPadding =
                      numElements + MemoryAllocator::template getPadding<typename Matrix::value_type>(numElements);
                    const uint32 maxCapacity = *maxCapacity_;

                    if (numElementsWithPadding > maxCapacity) {
                        const uint32 newCapacity = std::max<uint32>(maxCapacity * 2, numElementsWithPadding);
                        typename Matrix::value_type* array =
                          MemoryAllocator::template reallocateMemory<typename Matrix::value_type>(
                            this->array, maxCapacity, newCapacity);
                        (*maxCapacity_) = newCapacity;
                        this->array = array;
                        row_.array = array;
                    }

                    return numElementsWithPadding;
                }

                /**
                 * Inserts a new element at a specific position of the row.
                 *
                 * @param iterator  An iterator to the position where the new element should be inserted
                 * @param value     The value to be inserted at the given position
                 * @return          An iterator to the element that has been inserted
                 */
                Vector<typename Matrix::value_type>::iterator emplace(
                  Vector<typename Matrix::value_type>::iterator iterator, typename Matrix::value_type value) {
                    const uint32 numElements = this->numElements;
                    const uint32 newElements = numElements + 1;
                    const uint32 offset = iterator ? iterator - this->array : numElements;
                    uint32 newElementsWithPadding = this->reserve(newElements);
                    this->numElements = newElements;
                    row_.numElements = newElements;
                    this->numElementsWithPadding = newElementsWithPadding;
                    row_.numElementsWithPadding = newElementsWithPadding;
                    typename Matrix::value_type* array = this->array;

                    for (uint32 i = newElements - 1; i > offset; i--) {
                        array[i] = array[i - 1];
                    }

                    array[offset] = value;
                    return &array[offset];
                }

                /**
                 * Inserts a new elements at the end of the row.
                 *
                 * @param value The value to be inserted at the end of the row
                 * @return      An iterator to the element that has been inserted
                 */
                Vector<typename Matrix::value_type>::iterator emplace_back(typename Matrix::value_type value) {
                    return this->emplace(nullptr, value);
                }

                /**
                 * Removes the element at a specific position from the row.
                 *
                 * @param iterator  An iterator to the element that should be removed
                 * @return          An iterator to the new element at the given position
                 */
                Vector<typename Matrix::value_type>::iterator erase(
                  Vector<typename Matrix::value_type>::iterator iterator) {
                    const uint32 numElements = this->numElements;
                    const uint32 newElements = numElements - 1;
                    typename Matrix::value_type* array = this->array;
                    const uint32 offset = iterator - array;

                    for (uint32 i = offset; i < newElements; i++) {
                        array[i] = array[i + 1];
                    }

                    this->numElements = newElements;
                    row_.numElements = newElements;
                    const uint32 newElementsWithPadding =
                      newElements + MemoryAllocator::template getPadding<typename Matrix::value_type>(newElements);
                    this->numElementsWithPadding = newElementsWithPadding;
                    row_.numElementsWithPadding = newElementsWithPadding;
                    return &array[offset];
                }

                /**
                 * Removes all elements from the row.
                 */
                void clear() {
                    this->numElements = 0;
                    row_.numElements = 0;
                    this->numElementsWithPadding = 0;
                    row_.numElementsWithPadding = 0;
                }
        };

        /**
         * @param numRows   The number of rows in the view
         * @param numCols   The number of columns in the view
         */
        ListOfListsAllocator(uint32 numRows, uint32 numCols)
            : Matrix(new typename Matrix::Row[numRows] {}, numRows, numCols),
              maxCapacities_(DefaultMemoryAllocator::allocateMemory<uint32>(numRows, true)) {}

        /**
         * @param other A reference to an object of type `ListOfListsAllocator` that should be copied
         */
        ListOfListsAllocator(const ListOfListsAllocator<Matrix, MemoryAllocator>& other) = delete;

        /**
         * @param other A reference to an object of type `ListOfListsAllocator` that should be moved
         */
        ListOfListsAllocator(ListOfListsAllocator<Matrix, MemoryAllocator>&& other)
            : Matrix(std::move(other)), maxCapacities_(other.maxCapacities_) {
            other.release();
            other.maxCapacities_ = nullptr;
        }

        virtual ~ListOfListsAllocator() override {
            if (Matrix::array) {
                for (uint32 i = 0; i < this->numRows; i++) {
                    typename Matrix::Row& row = Matrix::array[i];

                    if (row.array) {
                        MemoryAllocator::freeMemory(row.array);
                    }
                }

                delete[] Matrix::array;
            }

            if (maxCapacities_) {
                DefaultMemoryAllocator::freeMemory(maxCapacities_);
            }
        }

        /**
         * Provides read-only access to an individual row in the view.
         */
        using const_row = Matrix::const_row;

        /**
         * Provides access to an individual row in the view and allows to modify it.
         */
        using row = Row;

        /**
         * Returns a view that provides read-only access to a specific row in the view.
         *
         * @param row   The index of the row
         * @return      A `const_row`
         */
        const_row operator[](uint32 row) const {
            return Matrix::operator[](row);
        }

        /**
         * Returns a view that provides access to a specific row in the view and allows to modify it.
         *
         * @param row   The index of the row
         * @return      A `row`
         */
        row operator[](uint32 row) {
            return Row(Matrix::array[row], &maxCapacities_[row]);
        }
};

/**
 * Allocates the memory, a `ListOfLists` provides access to.
 *
 * @tparam T                The type of the values stored in the `ListOfLists`
 * @tparam MemoryAllocator  The type of the memory allocator to be used
 */
template<typename T, typename MemoryAllocator = DefaultMemoryAllocator>
using AllocatedListOfLists = ListOfListsAllocator<ListOfLists<T>, MemoryAllocator>;

/**
 * Provides row-wise read and write access via iterators to the values stored in a sparse matrix in the list of lists
 * (LIL) format.
 *
 * @tparam Matrix The type of the matrix
 */
template<typename Matrix>
class MLRLCOMMON_API IterableListOfListsDecorator : public Matrix {
    public:

        /**
         * @param view The view, the matrix should be backed by
         */
        explicit IterableListOfListsDecorator(typename Matrix::view_type&& view) : Matrix(std::move(view)) {}

        virtual ~IterableListOfListsDecorator() override {}

        /**
         * An iterator that provides read-only access to the values in the matrix.
         */
        using value_const_iterator = Matrix::view_type::value_const_iterator;

        /**
         * An iterator that provides access to the values in the matrix and allows to modify them.
         */
        using value_iterator = Matrix::view_type::value_iterator;

        /**
         * Provides read-only access to an individual row in the matrix.
         */
        using const_row = Matrix::view_type::const_row;

        /**
         * Provides access to an individual row in the matrix and allows to modify it.
         */
        using row = Matrix::view_type::row;

        /**
         * Returns a view that provides read-only access to a specific row in the matrix.
         *
         * @param row   The index of the row
         * @return      A `const_row`
         */
        const_row operator[](uint32 row) const {
            return Matrix::view[row];
        }

        /**
         * Returns a view that provides access to a specific row in the matrix and allows to modify it.
         *
         * @param row   The index of the row
         * @return      A `row`
         */
        row operator[](uint32 row) {
            return Matrix::view[row];
        }

        /**
         * Returns a `value_const_iterator` to the beginning of the values in a specific row of the matrix.
         *
         * @param row   The index of the row
         * @return      A `value_const_iterator` to the beginning of the values
         */
        value_const_iterator values_cbegin(uint32 row) const {
            return Matrix::view.values_cbegin(row);
        }

        /**
         * Returns a `value_const_iterator` to the end of the values in a specific row of the matrix.
         *
         * @param row   The index of the row
         * @return      A `value_const_iterator` to the end of the values
         */
        value_const_iterator values_cend(uint32 row) const {
            return Matrix::view.values_cend(row);
        }

        /**
         * Returns a `value_iterator` to the beginning of the values in a specific row of the matrix.
         *
         * @param row   The index of the row
         * @return      A `value_iterator` to the beginning of the values
         */
        value_iterator values_begin(uint32 row) {
            return Matrix::view.values_begin(row);
        }

        /**
         * Returns a `value_iterator` to the end of the values in a specific row of the matrix.
         *
         * @param row   The index of the row
         * @return      A `value_iterator` to the end of the values
         */
        value_iterator values_end(uint32 row) {
            return Matrix::view.values_end(row);
        }
};
