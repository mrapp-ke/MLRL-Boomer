/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/view_matrix.hpp"

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
            return &Matrix::view.array[row * Matrix::view.numCols];
        }

        /**
         * Returns a `value_const_iterator` to the end of a specific row in the matrix.
         *
         * @param row   The index of the row
         * @return      A `value_const_iterator` to the end of the row
         */
        value_const_iterator values_cend(uint32 row) const {
            return &Matrix::view.array[(row + 1) * Matrix::view.numCols];
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
            return &Matrix::view.array[row * Matrix::view.numCols];
        }

        /**
         * Returns a `value_iterator` to the end of a specific row in the matrix.
         *
         * @param row   The index of the row
         * @return      A `value_iterator` to the end of the row
         */
        value_iterator values_end(uint32 row) {
            return &Matrix::view.array[(row + 1) * Matrix::view.numCols];
        }
};
