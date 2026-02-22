/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/view_matrix_dense.hpp"
#include "mlrl/common/data/view_vector.hpp"

#include <utility>

/**
 * A two-dimensional view that provides row-wise access to values stored in a pre-allocated array of a specific size.
 *
 * @tparam T The type of the values, the view provides access to
 */
template<typename T>
class MLRLCOMMON_API CContiguousView : public DenseMatrix<T> {
    public:

        /**
         * @param array     A pointer to an array of template type `T` that stores the values, the view should provide
         *                  access to
         * @param numRows   The number of rows in the view
         * @param numCols   The number of columns in the view
         */
        CContiguousView(T* array, uint32 numRows, uint32 numCols) : DenseMatrix<T>(array, numRows, numCols) {}

        /**
         * @param other A const reference to an object of type `CContiguousView` that should be copied
         */
        CContiguousView(const CContiguousView<T>& other) : DenseMatrix<T>(other) {}

        /**
         * @param other A reference to an object of type `CContiguousView` that should be moved
         */
        CContiguousView(CContiguousView<T>&& other) : DenseMatrix<T>(std::move(other)) {}

        virtual ~CContiguousView() override {}

        /**
         * Provides read-only access to an individual row in the view.
         */
        using const_row = const Vector<const T>;

        /**
         * Provides access to an individual row in the view and allows to modify it.
         */
        using row = Vector<T>;

        /**
         * Creates and returns a view that provides read-only access to a specific row in the view.
         *
         * @param row   The index of the row
         * @return      An object of type `const_row` that has been created
         */
        const_row operator[](uint32 row) const {
            return Vector<const T>(values_cbegin(row), Matrix::numCols);
        }

        /**
         * Creates and returns a view that provides access to a specific row in the view and allows to modify it.
         *
         * @param row   The index of the row
         * @return      An object of type `row` that has been created
         */
        row operator[](uint32 row) {
            return Vector<T>(values_begin(row), Matrix::numCols);
        }

        /**
         * Returns a `value_const_iterator` to the beginning of a specific row in the view.
         *
         * @param row   The index of the row
         * @return      A `value_const_iterator` to the beginning of the row
         */
        typename DenseMatrix<T>::value_const_iterator values_cbegin(uint32 row) const {
            return &DenseMatrix<T>::array[row * Matrix::numCols];
        }

        /**
         * Returns a `value_const_iterator` to the end of a specific row in the view.
         *
         * @param row   The index of the row
         * @return      A `value_const_iterator` to the end of the row
         */
        typename DenseMatrix<T>::value_const_iterator values_cend(uint32 row) const {
            return &DenseMatrix<T>::array[(row + 1) * Matrix::numCols];
        }

        /**
         * Returns a `value_iterator` to the beginning of a specific row in the view.
         *
         * @param row   The index of the row
         * @return      A `value_iterator` to the beginning of the row
         */
        typename DenseMatrix<T>::value_iterator values_begin(uint32 row) {
            return &DenseMatrix<T>::array[row * Matrix::numCols];
        }

        /**
         * Returns a `value_iterator` to the end of a specific row in the view.
         *
         * @param row   The index of the row
         * @return      A `value_iterator` to the end of the row
         */
        typename DenseMatrix<T>::value_iterator values_end(uint32 row) {
            return &DenseMatrix<T>::array[(row + 1) * Matrix::numCols];
        }
};

/**
 * Allocates the memory, a `CContiguousView` provides access to
 *
 * @tparam T The type of the values stored in the `CContiguousView`
 */
template<typename T>
using AllocatedCContiguousView = DenseMatrixAllocator<CContiguousView<T>>;
