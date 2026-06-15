/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/view_matrix.hpp"

#include <algorithm>
#include <utility>

/**
 * A two-dimensional view that provides access to values stored in a dense matrix of a specific size.
 *
 * @tparam T The type of the values, the view provides access to
 */
template<typename T>
class MLRLCOMMON_API DenseMatrix : public View<T>,
                                   public Matrix {
    public:

        /**
         * @param array     A pointer to an array of template type `T` that stores the values, the view should provide
         *                  access to
         * @param numRows   The number of rows in the view
         * @param numCols   The number of columns in the view
         * @param padding   The number of unused elements to be inserted at the end of each row or column
         */
        DenseMatrix(T* array, uint32 numRows, uint32 numCols, uint32 padding = 0)
            : View<T>(array, 0, padding), Matrix(numRows, numCols) {}

        /**
         * @param other A const reference to an object of type `DenseMatrix` that should be copied
         */
        DenseMatrix(const DenseMatrix<T>& other) : View<T>(other), Matrix(std::move(other)) {}

        /**
         * @param other A reference to an object of type `DenseMatrix` that should be moved
         */
        DenseMatrix(DenseMatrix<T>&& other) : View<T>(std::move(other)), Matrix(std::move(other)) {}

        virtual ~DenseMatrix() override {}

        /**
         * An iterator that provides read-only access to the values in the view.
         */
        using value_const_iterator = const View<T>::value_type*;

        /**
         * An iterator that provides access to the values in the view and allows to modify them.
         */
        using value_iterator = View<T>::value_type*;
};

/**
 * Provides row- or column-wise access via iterators to the values stored in a dense matrix.
 *
 * @tparam Matrix The type of the matrix
 */
template<typename Matrix>
class MLRLCOMMON_API IterableDenseMatrixDecorator : public Matrix {
    public:

        /**
         * @param view The view, the matrix should be backed by
         */
        explicit IterableDenseMatrixDecorator(typename Matrix::view_type&& view) : Matrix(std::move(view)) {}

        virtual ~IterableDenseMatrixDecorator() override {}

        /**
         * An iterator that provides read-only access to the values stored in the matrix.
         */
        using value_const_iterator = Matrix::view_type::value_const_iterator;

        /**
         * An iterator that provides access to the values stored in the matrix and allows to modify them.
         */
        using value_iterator = Matrix::view_type::value_iterator;

        /**
         * Returns a `value_const_iterator` to the beginning of a specific row or column in the matrix, depending on the
         * memory layout of the view, the matrix is backed by.
         *
         * @param index The index of the row or column
         * @return      A `value_const_iterator` to the beginning of the row or column
         */
        value_const_iterator values_cbegin(uint32 index) const {
            return Matrix::view.values_cbegin(index);
        }

        /**
         * Returns a `value_const_iterator` to the end of a specific row or column in the matrix, depending on the
         * memory layout of the view, the matrix is backed by.
         *
         * @param index The index of the row or column
         * @return      A `value_const_iterator` to the end of the row or column
         */
        value_const_iterator values_cend(uint32 index) const {
            return Matrix::view.values_cend(index);
        }

        /**
         * Returns a `value_iterator` to the beginning of a specific row or column in the matrix, depending on the
         * memory layout of the view, the memory is backed by.
         *
         * @param index The index of the row or column
         * @return      A `value_iterator` to the beginning of the row or column
         */
        value_iterator values_begin(uint32 index) {
            return Matrix::view.values_begin(index);
        }

        /**
         * Returns a `value_iterator` to the end of a specific row or column in the matrix, depending on the memory
         * layout of the view, the memory is backed by.
         *
         * @param index The index of the row or column
         * @return      A `value_iterator` to the end of the row or column
         */
        value_iterator values_end(uint32 index) {
            return Matrix::view.values_end(index);
        }
};
