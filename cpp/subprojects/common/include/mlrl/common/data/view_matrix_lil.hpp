/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/view_matrix.hpp"

#include <vector>

/**
 * A two-dimensional view that provides row-wise access to values stored in a sparse matrix in the list of lists (LIL)
 * format.
 *
 * @tparam T The type
 */
template<typename T>
class ListOfLists : public Matrix {
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
        typedef T value_type;

        /**
         * An iterator that provides read-only access to the values in the view.
         */
        typedef typename std::vector<T>::const_iterator const_iterator;

        /**
         * An iterator that provides access to the values in the view and allows to modify them.
         */
        typedef typename std::vector<T>::iterator iterator;

        /**
         * Provides read-only access to an individual row in the view.
         */
        typedef const typename std::vector<T>& const_row;

        /**
         * Provides access to an individual row in the view and allows to modify it.
         */
        typedef typename std::vector<T>& row;

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
         * Returns a `const_iterator` to the beginning of the values in a specific row of the matrix.
         *
         * @param row   The index of the row
         * @return      A `const_iterator` to the beginning of the values
         */
        const_iterator cbegin(uint32 row) const {
            return array[row].cbegin();
        }

        /**
         * Returns a `const_iterator` to the end of the values in a specific row of the matrix.
         *
         * @param row   The index of the row
         * @return      A `const_iterator` to the end of the values
         */
        const_iterator cend(uint32 row) const {
            return array[row].cend();
        }

        /**
         * Returns an `iterator` to the beginning of the values in a specific row of the matrix.
         *
         * @param row   The index of the row
         * @return      An `iterator` to the beginning of the values
         */
        iterator begin(uint32 row) {
            return array[row].begin();
        }

        /**
         * Returns an `iterator` to the end of the values in a specific row of the matrix.
         *
         * @param row   The index of the row
         * @return      An `iterator` to the end of the values
         */
        iterator end(uint32 row) {
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
class ListOfListsAllocator : public Matrix {
    public:

        /**
         * @param numRows   The number of rows in the view
         * @param numCols   The number of columns in the view
         */
        ListOfListsAllocator(uint32 numRows, uint32 numCols)
            : Matrix(
              new std::vector<typename Matrix::value_type>[numRows] {
        },
              numRows, numCols) {}

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
