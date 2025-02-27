/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/view_matrix_c_contiguous.hpp"
#include "mlrl/common/data/view_matrix_composite.hpp"
#include "mlrl/common/data/view_matrix_lil.hpp"
#include "mlrl/common/data/view_vector_sparse_set.hpp"

#include <utility>
#include <vector>

/**
 * A two-dimensional view that provides random read and write access, as well as row-wise read and write access via
 * iterators, to values stored in a sparse matrix in the list of lists (LIL) format. Compared to the view
 * `ListOfLists`, the ability to provide random access to the elements in the view comes at the expense of memory
 * efficiency, as it does not only require to maintain a sparse matrix, but also a dense matrix that stores for each
 * element the corresponding position in the sparse matrix, if available.
 *
 * The data structure that is used for the representation of a single row is often referred to as an "unordered sparse
 * set". It was originally proposed in "An efficient representation for sparse sets", Briggs, Torczon, 1993 (see
 * https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=c83ae968c51219db68a03fc5b01de751dd2fe9ae).
 *
 * @tparam T The type of the values, the view provides access to
 */
template<typename T>
class MLRLCOMMON_API SparseSetView
    : public CompositeMatrix<AllocatedListOfLists<IndexedValue<T>>, AllocatedCContiguousView<uint32>> {
    public:

        /**
         * The index that is used to indicate that the value at a specific row and column is zero.
         */
        static inline constexpr uint32 MAX_INDEX = SparseSetVector<T>::MAX_INDEX;

        /**
         * @param numRows   The number of rows in the view
         * @param numCols   The number of columns in the view
         */
        SparseSetView(uint32 numRows, uint32 numCols)
            : CompositeMatrix<AllocatedListOfLists<IndexedValue<T>>, AllocatedCContiguousView<uint32>>(
                AllocatedListOfLists<IndexedValue<T>>(numRows, numCols),
                AllocatedCContiguousView<uint32>(numRows, numCols), numRows, numCols) {
            util::setViewToValue(this->secondView.array, this->secondView.numRows * this->secondView.numCols,
                                 MAX_INDEX);
        }

        /**
         * @param other A reference to an object of type `SparseSetView` that should be moved
         */
        SparseSetView(SparseSetView&& other)
            : CompositeMatrix<AllocatedListOfLists<IndexedValue<T>>, AllocatedCContiguousView<uint32>>(
                std::move(other)) {}

        virtual ~SparseSetView() override {}

        /**
         * The type of the values, the view provides access to.
         */
        typedef typename AllocatedListOfLists<IndexedValue<T>>::value_type value_type;

        /**
         * An iterator that provides read-only access to the values in the view.
         */
        typedef typename AllocatedListOfLists<IndexedValue<T>>::value_const_iterator value_const_iterator;

        /**
         * An iterator that provides access to the values in the view and allows to modify them.
         */
        typedef typename AllocatedListOfLists<IndexedValue<T>>::value_iterator value_iterator;

        /**
         * Provides read-only access to an individual row in the view.
         */
        typedef const SparseSetVector<T, const std::vector<IndexedValue<T>>, const uint32> const_row;

        /**
         * Provides access to an individual row in the view and allows to modify it.
         */
        typedef SparseSetVector<T> row;

        /**
         * Creates and returns a view that provides read-only access to a specific row in the view.
         *
         * @param row   The index of the row
         * @return      A `const_row`
         */
        const_row operator[](uint32 row) const {
            return SparseSetView<T>::const_row(&this->firstView[row], this->secondView.values_cbegin(row),
                                               this->numCols);
        }

        /**
         * Creates and returns a view that provides access to a specific row in the view and allows to modify it.
         *
         * @param row   The index of the row
         * @return      A `row`
         */
        row operator[](uint32 row) {
            return SparseSetView<T>::row(&this->firstView[row], this->secondView.values_begin(row), this->numCols);
        }

        /**
         * Returns a `value_const_iterator` to the beginning of the values in a specific row of the view.
         *
         * @param row   The index of the row
         * @return      A `value_const_iterator` to the beginning of the values
         */
        value_const_iterator values_cbegin(uint32 row) const {
            return this->firstView.values_cbegin(row);
        }

        /**
         * Returns a `value_const_iterator` to the end of the values in a specific row of the view.
         *
         * @param row   The index of the row
         * @return      A `value_const_iterator` to the end of the values
         */
        value_const_iterator values_cend(uint32 row) const {
            return this->firstView.values_cend(row);
        }

        /**
         * Returns a `value_iterator` to the beginning of the values in a specific row of the view.
         *
         * @param row   The index of the row
         * @return      A `value_iterator` to the beginning of the values
         */
        value_iterator values_begin(uint32 row) {
            return this->firstView.values_begin(row);
        }

        /**
         * Returns a `value_iterator` to the end of the values in a specific row of the view.
         *
         * @param row   The index of the row
         * @return      A `value_iterator` to the end of the values
         */
        value_iterator values_end(uint32 row) {
            return this->firstView.values_end(row);
        }

        /**
         * Sets all values stored in the matrix to zero.
         */
        void clear() {
            for (uint32 i = 0; i < Matrix::numRows; i++) {
                (*this)[i].clear();
            }
        }
};

/**
 * Provides random read and write access, as well as row-wise read and write access via iterators, to values stored in a
 * sparse matrix in the list of lists (LIL) format.
 *
 * @tparam Matrix The type of the matrix
 */
template<typename Matrix>
class MLRLCOMMON_API IterableSparseSetViewDecorator : public Matrix {
    public:

        /**
         * @param view The view, the matrix should be backed by
         */
        explicit IterableSparseSetViewDecorator(typename Matrix::view_type&& view) : Matrix(std::move(view)) {}

        virtual ~IterableSparseSetViewDecorator() override {}

        /**
         * An iterator that provides read-only access to the values in the matrix.
         */
        typedef typename Matrix::view_type::value_const_iterator value_const_iterator;

        /**
         * An iterator that provides access to the values in the matrix and allows to modify them.
         */
        typedef typename Matrix::view_type::value_iterator value_iterator;

        /**
         * Provides read-only access to an individual row in the matrix.
         */
        typedef typename Matrix::view_type::const_row const_row;

        /**
         * Provides access to an individual row in the matrix and allows to modify it.
         */
        typedef typename Matrix::view_type::row row;

        /**
         * Creates and returns a view that provides read-only access to a specific row in the matrix.
         *
         * @param row   The index of the row
         * @return      A `const_row`
         */
        const_row operator[](uint32 row) const {
            return Matrix::view[row];
        }

        /**
         * Creates and returns a view that provides access to a specific row in the matrix and allows to modify it.
         *
         * @param row   The index of the row
         * @return      A `row`
         */
        row operator[](uint32 row) {
            return Matrix::view[row];
        }

        /**
         * Returns a `value_const_iterator` to the beginning of the values in a specific row of the matrix.
         *
         * @param row   The index of the row
         * @return      A `value_const_iterator` to the beginning of the values
         */
        value_const_iterator values_cbegin(uint32 row) const {
            return Matrix::view.values_cbegin(row);
        }

        /**
         * Returns a `value_const_iterator` to the end of the values in a specific row of the matrix.
         *
         * @param row   The index of the row
         * @return      A `value_const_iterator` to the end of the values
         */
        value_const_iterator values_cend(uint32 row) const {
            return Matrix::view.values_cend(row);
        }

        /**
         * Returns a `value_iterator` to the beginning of the values in a specific row of the matrix.
         *
         * @param row   The index of the row
         * @return      A `value_iterator` to the beginning of the values
         */
        value_iterator values_begin(uint32 row) {
            return Matrix::view.values_begin(row);
        }

        /**
         * Returns a `value_iterator` to the end of the values in a specific row of the matrix.
         *
         * @param row   The index of the row
         * @return      A `value_iterator` to the end of the values
         */
        value_iterator values_end(uint32 row) {
            return Matrix::view.values_end(row);
        }
};
