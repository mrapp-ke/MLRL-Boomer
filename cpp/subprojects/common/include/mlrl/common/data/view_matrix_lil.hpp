/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/view_matrix.hpp"

#include <utility>
#include <vector>

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
         * A pointer to an array that stores vectors corresponding to each row in the matrix.
         */
        std::vector<T>* array;

        /**
         * @param array     A pointer to an array of type `std::vector`, shape `(numRows)`, that stores vectors
         *                  corresponding to each row in the matrix
         * @param numRows   The number of rows in the view
         * @param numCols   The number of columns in the view
         */
        ListOfLists(std::vector<T>* array, uint32 numRows, uint32 numCols) : Matrix(numRows, numCols), array(array) {}

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
        using value_type = T;

        /**
         * An iterator that provides read-only access to the values in the view.
         */
        using value_const_iterator = std::vector<T>::const_iterator;

        /**
         * An iterator that provides access to the values in the view and allows to modify them.
         */
        using value_iterator = std::vector<T>::iterator;

        /**
         * Provides read-only access to an individual row in the view.
         */
        using const_row = const std::vector<T>&;

        /**
         * Provides access to an individual row in the view and allows to modify it.
         */
        using row = std::vector<T>&;

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
         * Releases the ownership of the array that stores vectors corresponding to each row in the matrix. As a result,
         * the behavior of this view becomes undefined and it should not be used anymore. The caller is responsible for
         * freeing the memory that is occupied by the array.
         *
         * @return A pointer to the array that stores vectors corresponding to each row in the matrix
         */
        std::vector<T>* release() {
            std::vector<T>* ptr = array;
            array = nullptr;
            return ptr;
        }
};

/**
 * Allocates the memory for a two-dimensional view that provides row-wise access to values stored in a matrix in the
 * list of lists (LIL) format.
 *
 * @tparam Matrix The type of the view
 */
template<typename Matrix>
class MLRLCOMMON_API ListOfListsAllocator : public Matrix {
    public:

        /**
         * @param numRows   The number of rows in the view
         * @param numCols   The number of columns in the view
         */
        ListOfListsAllocator(uint32 numRows, uint32 numCols)
            : Matrix(new std::vector<typename Matrix::value_type>[numRows] {
              }, numRows, numCols) {}

        /**
         * @param other A reference to an object of type `ListOfListsAllocator` that should be copied
         */
        ListOfListsAllocator(const ListOfListsAllocator<Matrix>& other) : Matrix(other) {
            throw std::runtime_error("Objects of type ListOfListsAllocator cannot be copied");
        }

        /**
         * @param other A reference to an object of type `ListOfListsAllocator` that should be moved
         */
        ListOfListsAllocator(ListOfListsAllocator<Matrix>&& other) : Matrix(std::move(other)) {
            other.release();
        }

        virtual ~ListOfListsAllocator() override {
            if (Matrix::array) {
                delete[] Matrix::array;
            }
        }
};

/**
 * Allocates the memory, a `ListOfLists` provides access to.
 *
 * @tparam T The type of the values stored in the `ListOfLists`
 */
template<typename T>
using AllocatedListOfLists = ListOfListsAllocator<ListOfLists<T>>;

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
