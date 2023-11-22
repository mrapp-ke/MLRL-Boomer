/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/view_matrix.hpp"

/**
 * A two-dimensional view that provides row-wise access to values stored in a pre-allocated array of a specific size.
 *
 * @tparam T The type of the values, the view provides access to
 */
template<typename T>
class CContiguousView : public Matrix<T> {
    public:

        /**
         * @param array     A pointer to an array of template type `T` that stores the values, the view should provide
         *                  access to
         * @param numRows   The number of rows in the view
         * @param numCols   The number of columns in the view
         */
        CContiguousView(T* array, uint32 numRows, uint32 numCols) : Matrix<T>(array, numRows, numCols) {}

        /**
         * @param other A const reference to an object of type `CContiguousView` that should be copied
         */
        CContiguousView(const CContiguousView<T>& other) : Matrix<T>(other.array, other.numRows, other.numCols) {}

        /**
         * @param other A reference to an object of type `CContiguousView` that should be moved
         */
        CContiguousView(CContiguousView<T>&& other) : Matrix<T>(other.array, other.numRows, other.numCols) {}

        virtual ~CContiguousView() override {}

        /**
         * Returns a `value_const_iterator` to the beginning of a specific row in the view.
         *
         * @param row   The index of the row
         * @return      A `value_const_iterator` to the beginning of the row
         */
        typename DenseMatrix<T>::value_const_iterator values_cbegin(uint32 row) const {
            return &View<T>::array[row * Matrix<T>::numCols];
        }

        /**
         * Returns a `value_const_iterator` to the end of a specific row in the view.
         *
         * @param row   The index of the row
         * @return      A `value_const_iterator` to the end of the row
         */
        typename DenseMatrix<T>::const_iterator values_cend(uint32 row) const {
            return &View<T>::array[(row + 1) * Matrix<T>::numCols];
        }

        /**
         * Returns a `value_iterator` to the beginning of a specific row in the view.
         *
         * @param row   The index of the row
         * @return      A `value_iterator` to the beginning of the row
         */
        typename DenseMatrix<T>::value_iterator values_begin(uint32 row) {
            return &View<T>::array[row * Matrix<T>::numCols];
        }

        /**
         * Returns a `value_iterator` to the end of a specific row in the view.
         *
         * @param row   The index of the row
         * @return      A `value_iterator` to the end of the row
         */
        typename DenseMatrix<T>::value_iterator values_end(uint32 row) {
            return &View<T>::array[(row + 1) * Matrix<T>::numCols];
        }
};

/**
 * Allocates the memory, a `CContiguousView` provides access to
 *
 * @tparam T The type of the values stored in the `CContiguousView`
 */
template<typename T>
using AllocatedCContiguousView = MatrixAllocator<CContiguousView<T>>;

/**
 * Provides row-wise read-only access via iterators to the values stored in a C-contiguous matrix.
 *
 * @tparam Matrix The type of the matrix
 */
template<typename Matrix>
class ReadIterableCContiguousMatrixDecorator : public Matrix {
    public:

        /**
         * @param view The view, the matrix should be backed by
         */
        ReadIterableCContiguousMatrixDecorator(typename Matrix::view_type&& view) : Matrix(std::move(view)) {}

        virtual ~ReadIterableCContiguousMatrixDecorator() override {}

        /**
         * An iterator that provides read-only access to the values stored in the matrix.
         */
        typedef typename Matrix::view_type::const_iterator value_const_iterator;

        /**
         * Returns a `value_const_iterator` to the beginning of a specific row in the matrix.
         *
         * @param row   The index of the row
         * @return      A `value_const_iterator` to the beginning of the row
         */
        value_const_iterator values_cbegin(uint32 row) const {
            return &Matrix::view.values_cbegin(row);
        }

        /**
         * Returns a `value_const_iterator` to the end of a specific row in the matrix.
         *
         * @param row   The index of the row
         * @return      A `value_const_iterator` to the end of the row
         */
        value_const_iterator values_cend(uint32 row) const {
            return &Matrix::view.values_cend(row);
        }
};

/**
 * Provides row-wise write access via iterators to the values stored in a C-contiguous matrix.
 *
 * @tparam Matrix The type of the matrix
 */
template<typename Matrix>
class WriteIterableCContiguousMatrixDecorator : public Matrix {
    public:

        /**
         * @param view The view, the matrix should be backed by
         */
        WriteIterableCContiguousMatrixDecorator(typename Matrix::view_type&& view) : Matrix(std::move(view)) {}

        virtual ~WriteIterableCContiguousMatrixDecorator() override {}

        /**
         * An iterator that provides access to the values stored in the matrix and allows to modify them.
         */
        typedef typename Matrix::view_type::iterator value_iterator;

        /**
         * Returns a `value_iterator` to the beginning of a specific row in the matrix.
         *
         * @param row   The index of the row
         * @return      A `value_iterator` to the beginning of the row
         */
        value_iterator values_begin(uint32 row) {
            return &Matrix::view.values_begin(row);
        }

        /**
         * Returns a `value_iterator` to the end of a specific row in the matrix.
         *
         * @param row   The index of the row
         * @return      A `value_iterator` to the end of the row
         */
        value_iterator values_end(uint32 row) {
            return &Matrix::view.values_end(row);
        }
};

/**
 * Provides read-only access via iterators to the values stored in a C-contiguous matrix.
 *
 * @tparam Matrix The type of the matrix
 */
template<typename Matrix>
using ReadableCContiguousMatrixDecorator = ReadIterableCContiguousMatrixDecorator<MatrixDecorator<Matrix>>;

/**
 * Provides read and write access via iterators to the values stored in a C-contiguous matrix.
 *
 * @tparam Matrix The type of the matrix
 */
template<typename Matrix>
using WritableCContiguousMatrixDecorator =
  WriteIterableCContiguousMatrixDecorator<ReadableCContiguousMatrixDecorator<Matrix>>;
