/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/view_matrix.hpp"

/**
 * A two-dimensional view that provides access to values stored in a dense matrix of a specific size.
 *
 * @tparam T The type of the values, the view provides access to
 */
template<typename T>
class DenseMatrix : public Matrix<T> {
    public:

        /**
         * @param array     A pointer to an array of template type `T` that stores the values, the view should provide
         *                  access to
         * @param numRows   The number of rows in the view
         * @param numCols   The number of columns in the view
         */
        DenseMatrix(T* array, uint32 numRows, uint32 numCols) : Matrix<T>(array, numRows, numCols) {}

        /**
         * @param other A const reference to an object of type `DenseMatrix` that should be copied
         */
        DenseMatrix(const DenseMatrix<T>& other) : Matrix<T>(other.array, other.numRows, other.numCols) {}

        /**
         * @param other A reference to an object of type `DenseMatrix` that should be moved
         */
        DenseMatrix(DenseMatrix<T>&& other) : Matrix<T>(other.array, other.numRows, other.numCols) {}

        virtual ~DenseMatrix() override {}

        /**
         * Returns a `const_iterator` to the end of the view.
         *
         * @return A `const_iterator` to the end
         */
        typename View<T>::const_iterator cend() const {
            return &View<T>::array[Matrix<T>::numRows * Matrix<T>::numCols];
        }

        /**
         * Returns an `iterator` to the end of the view.
         *
         * @return An `iterator` to the end
         */
        typename View<T>::iterator end() {
            return &View<T>::array[Matrix<T>::numRows * Matrix<T>::numCols];
        }
};

/**
 * Allocates the memory, a two-dimensional dense view provides access to.
 *
 * @tparam Matrix The type of the view
 */
template<typename Matrix>
class DenseMatrixAllocator : public Matrix {
    public:

        /**
         * @param numRows   The number of rows in the view
         * @param numCols   The number of columns in the view
         * @param init      True, if all elements in the view should be value-initialized, false otherwise
         */
        DenseMatrixAllocator(uint32 numRows, uint32 numCols, bool init = false)
            : Matrix(allocateMemory<typename Matrix::value_type>(numRows * numCols, init), numRows, numCols) {}

        /**
         * @param other A reference to an object of type `DenseMatrixAllocator` that should be moved
         */
        DenseMatrixAllocator(DenseMatrixAllocator<Matrix>&& other) : Matrix(std::move(other)) {
            other.array = nullptr;
        }

        virtual ~DenseMatrixAllocator() override {
            if (Matrix::array) {
                freeMemory(Matrix::array);
            }
        }
};

/**
 * Provides row- or column-wise access via iterators to the values stored in a dense matrix.
 *
 * @tparam Matrix The type of the matrix
 */
template<typename Matrix>
class IterableDenseMatrixDecorator : public Matrix {
    public:

        /**
         * @param view The view, the matrix should be backed by
         */
        IterableDenseMatrixDecorator(typename Matrix::view_type&& view) : Matrix(std::move(view)) {}

        virtual ~IterableDenseMatrixDecorator() override {}

        /**
         * An iterator that provides read-only access to the values stored in the matrix.
         */
        typedef typename Matrix::view_type::const_iterator value_const_iterator;

        /**
         * An iterator that provides access to the values stored in the matrix and allows to modify them.
         */
        typedef typename Matrix::view_type::iterator value_iterator;

        /**
         * Returns a `value_const_iterator` to the beginning of a specific row or column in the matrix, depending on the
         * memory layout of the view, the matrix is backed by.
         *
         * @param index The index of the row or column
         * @return      A `value_const_iterator` to the beginning of the row or column
         */
        value_const_iterator values_cbegin(uint32 index) const {
            return &Matrix::view.values_cbegin(index);
        }

        /**
         * Returns a `value_const_iterator` to the end of a specific row or column in the matrix, depending on the
         * memory layout of the view, the matrix is backed by.
         *
         * @param index The index of the row or column
         * @return      A `value_const_iterator` to the end of the row or column
         */
        value_const_iterator values_cend(uint32 index) const {
            return &Matrix::view.values_cend(index);
        }

        /**
         * Returns a `value_iterator` to the beginning of a specific row or column in the matrix, depending on the
         * memory layout of the view, the memory is backed by.
         *
         * @param index The index of the row or column
         * @return      A `value_iterator` to the beginning of the row or column
         */
        value_iterator values_begin(uint32 index) {
            return &Matrix::view.values_begin(index);
        }

        /**
         * Returns a `value_iterator` to the end of a specific row or column in the matrix, depending on the memory
         * layout of the view, the memory is backed by.
         *
         * @param index The index of the row or column
         * @return      A `value_iterator` to the end of the row or column
         */
        value_iterator values_end(uint32 index) {
            return &Matrix::view.values_end(index);
        }
};