/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/view.hpp"

/**
 * A two-dimensional view that provides access to values stored in a pre-allocated array of a specific size.
 *
 * @tparam T The type of the values, the view provides access to
 */
template<typename T>
class Matrix : public View<T> {
    public:

        /**
         * The number of rows in the view.
         */
        uint32 numRows;

        /**
         * The number of columns in the view.
         */
        uint32 numCols;

        /**
         * @param array     A pointer to an array of template type `T` that stores the values, the view should provide
         *                  access to
         * @param numRows   The number of rows in the view
         * @param numCols   The number of columns in the view
         */
        Matrix(T* array, uint32 numRows, uint32 numCols) : View<T>(array), numRows(numRows), numCols(numCols) {}

        /**
         * @param other A const reference to an object of type `Matrix` that should be copied
         */
        Matrix(const Matrix<T>& other) : View<T>(other.array), numRows(other.numRows), numCols(other.numCols) {}

        /**
         * @param other A reference to an object of type `Matrix` that should be moved
         */
        Matrix(Matrix<T>&& other) : View<T>(other.array), numRows(other.numRows), numCols(other.numCols) {}

        virtual ~Matrix() override {}

        /**
         * Returns a `const_iterator` to the end of the view.
         *
         * @return A `const_iterator` to the end
         */
        typename View<T>::const_iterator cend() const {
            return &View<T>::array[numRows * numCols];
        }

        /**
         * Returns an `iterator` to the end of the view.
         *
         * @return An `iterator` to the end
         */
        typename View<T>::iterator end() {
            return &View<T>::array[numRows * numCols];
        }
};

/**
 * Allocates the memory, a two-dimensional view provides access to.
 *
 * @tparam View The type of the view
 */
template<typename Matrix>
class MatrixAllocator : public Matrix {
    public:

        /**
         * @param numRows   The number of rows in the view
         * @param numCols   The number of columns in the view
         * @param init      True, if all elements in the view should be value-initialized, false otherwise
         */
        MatrixAllocator(uint32 numRows, uint32 numCols, bool init = false)
            : Matrix(allocateMemory<typename Matrix::value_type>(numRows * numCols, init), numRows, numCols) {}

        /**
         * @param other A reference to an object of type `MatrixAllocator` that should be moved
         */
        MatrixAllocator(MatrixAllocator<Matrix>&& other) : Matrix(std::move(other)) {
            other.array = nullptr;
        }

        virtual ~MatrixAllocator() override {
            freeMemory(Matrix::array);
        }
};

/**
 * A matrix that is backed by a two-dimensional view of a specific size.
 *
 * @tparam View The type of view, the matrix is backed by
 */
template<typename View>
class MatrixDecorator : public ViewDecorator<View> {
    public:

        /**
         * @param view The view, the matrix should be backed by
         */
        MatrixDecorator(View&& view) : ViewDecorator<View>(std::move(view)) {}

        virtual ~MatrixDecorator() override {}

        /**
         * Returns the number of rows in the matrix.
         *
         * @return The number of rows in the matrix
         */
        uint32 getNumRows() const {
            return ViewDecorator<View>::view.numRows;
        }

        /**
         * Returns the number of columns in the matrix.
         *
         * @return The number of columns in the matrix
         */
        uint32 getNumCols() const {
            return ViewDecorator<View>::view.numCols;
        }
};
