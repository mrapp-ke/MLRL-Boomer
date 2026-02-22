/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/view_matrix_dense.hpp"
#include "mlrl/common/data/view_vector.hpp"

#include <utility>

/**
 * A two-dimensional view that provides column-wise access to values stored in a pre-allocated array of a specific size.
 *
 * @tparam T The type of the values, the view provides access to
 */
template<typename T>
class MLRLCOMMON_API FortranContiguousView : public DenseMatrix<T> {
    public:

        /**
         * @param array     A pointer to an array of template type `T` that stores the values, the view should provide
         *                  access to
         * @param numRows   The number of rows in the view
         * @param numCols   The number of columns in the view
         */
        FortranContiguousView(T* array, uint32 numRows, uint32 numCols) : DenseMatrix<T>(array, numRows, numCols) {}

        /**
         * @param other A const reference to an object of type `FortranContiguousView` that should be copied
         */
        FortranContiguousView(const FortranContiguousView<T>& other) : DenseMatrix<T>(other) {}

        /**
         * @param other A reference to an object of type `FortranContiguousView` that should be moved
         */
        FortranContiguousView(FortranContiguousView<T>&& other) : DenseMatrix<T>(std::move(other)) {}

        virtual ~FortranContiguousView() override {}

        /**
         * Provides read-only access to an individual column in the view.
         */
        using const_column = const Vector<const T>;

        /**
         * Provides access to an individual column in the view and allows to modify it.
         */
        using column = Vector<T>;

        /**
         * Creates and returns a view that provides read-only access to a specific column in the view.
         *
         * @param column    The index of the column
         * @return          An object of type `const_column` that has been created
         */
        const_column operator[](uint32 column) const {
            return Vector<const T>(values_cbegin(column), Matrix::numRows);
        }

        /**
         * Creates and returns a view that provides access to a specific column in the view and allows to modify it.
         *
         * @param column    The index of the column
         * @return          An object of type `column` that has been created
         */
        column operator[](uint32 column) {
            return Vector<T>(values_begin(column), Matrix::numRows);
        }

        /**
         * Returns a `value_const_iterator` to the beginning of a specific column in the view.
         *
         * @param column    The index of the column
         * @return          A `value_const_iterator` to the beginning of the column
         */
        typename DenseMatrix<T>::value_const_iterator values_cbegin(uint32 column) const {
            return &DenseMatrix<T>::array[column * Matrix::numRows];
        }

        /**
         * Returns a `value_const_iterator` to the end of a specific column in the view.
         *
         * @param column    The index of the column
         * @return          A `value_const_iterator` to the end of the column
         */
        typename DenseMatrix<T>::value_const_iterator values_cend(uint32 column) const {
            return &DenseMatrix<T>::array[(column + 1) * Matrix::numRows];
        }

        /**
         * Returns a `value_iterator` to the beginning of a specific column in the view.
         *
         * @param column    The index of the column
         * @return          A `value_iterator` to the beginning of the column
         */
        typename DenseMatrix<T>::value_iterator values_begin(uint32 column) {
            return &DenseMatrix<T>::array[column * Matrix::numRows];
        }

        /**
         * Returns a `value_iterator` to the end of a specific column in the view.
         *
         * @param column    The index of the column
         * @return          A `value_iterator` to the end of the column
         */
        typename DenseMatrix<T>::value_iterator values_end(uint32 column) {
            return &DenseMatrix<T>::array[(column + 1) * Matrix::numRows];
        }
};

/**
 * Allocates the memory, a `FortranContiguousView` provides access to
 *
 * @tparam T The type of the values stored in the `FortranContiguousView`
 */
template<typename T>
using AllocatedFortranContiguousView = DenseMatrixAllocator<FortranContiguousView<T>>;
