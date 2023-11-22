/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/view_matrix.hpp"

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
