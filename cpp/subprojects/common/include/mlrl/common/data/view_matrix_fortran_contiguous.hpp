/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/view_matrix.hpp"

/**
 * A two-dimensional view that provides column-wise access to values stored in a pre-allocated array of a specific size.
 *
 * @tparam T The type of the values, the view provides access to
 */
template<typename T>
class FortranContiguousView : public Matrix<T> {
    public:

        /**
         * @param array     A pointer to an array of template type `T` that stores the values, the view should provide
         *                  access to
         * @param numRows   The number of rows in the view
         * @param numCols   The number of columns in the view
         */
        FortranContiguousView(T* array, uint32 numRows, uint32 numCols) : Matrix<T>(array, numRows, numCols) {}

        /**
         * @param other A const reference to an object of type `FortranContiguousView` that should be copied
         */
        FortranContiguousView(const FortranContiguousView<T>& other)
            : Matrix<T>(other.array, other.numRows, other.numCols) {}

        /**
         * @param other A reference to an object of type `FortranContiguousView` that should be moved
         */
        FortranContiguousView(FortranContiguousView<T>&& other) : Matrix<T>(other.array, other.numRows, other.numCols) {}

        virtual ~FortranContiguousView() override {}

        /**
         * Returns a `value_const_iterator` to the beginning of a specific column in the view.
         *
         * @param column    The index of the column
         * @return          A `value_const_iterator` to the beginning of the column
         */
        typename View<T>::value_const_iterator values_cbegin(uint32 column) const {
            return &View<T>::array[column * Matrix<T>::numRows];
        }

        /**
         * Returns a `value_const_iterator` to the end of a specific column in the view.
         *
         * @param column    The index of the column
         * @return          A `value_const_iterator` to the end of the column
         */
        typename View<T>::const_iterator values_cend(uint32 column) const {
            return &View<T>::array[(column + 1) * Matrix<T>::numRows];
        }

        /**
         * Returns a `value_iterator` to the beginning of a specific column in the view.
         *
         * @param column    The index of the column
         * @return          A `value_iterator` to the beginning of the column
         */
        typename View<T>::value_iterator values_begin(uint32 column) {
            return &View<T>::array[column * Matrix<T>::numRows];
        }

        /**
         * Returns a `value_iterator` to the end of a specific column in the view.
         *
         * @param column    The index of the column
         * @return          A `value_iterator` to the end of the column
         */
        typename View<T>::value_iterator values_end(uint32 column) {
            return &View<T>::array[(column + 1) * Matrix<T>::numRows];
        }
};
