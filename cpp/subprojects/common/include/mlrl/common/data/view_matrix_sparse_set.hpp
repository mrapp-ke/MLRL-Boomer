/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/indexed_value.hpp"
#include "mlrl/common/data/view_matrix_c_contiguous.hpp"
#include "mlrl/common/data/view_matrix_composite.hpp"
#include "mlrl/common/data/view_matrix_lil.hpp"

#include <limits>
#include <utility>

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

    private:

        /**
         * Provides read-only access to a single row of a `SparseSetView`.
         *
         * @tparam ValueRow         The type of the object that provides access to the values of all dense elements
         *                          explicitly stored in the row
         * @tparam IndexIterator    The type of the iterator that provides access to the column indices of all dense
         *                          elements explicitly stored in the row
         */
        template<typename ValueRow, typename IndexIterator>
        class ConstRow {
            protected:

                /**
                 * A view that provides access to all dense elements explicitly stored in the row.
                 */
                ValueRow row_;

                /**
                 * A view that provides access to the column indices of all dense elements explicitly stored in the row.
                 */
                IndexIterator indexIterator_;

            public:

                /**
                 * The number of dense elements explicitly stored in the row.
                 */
                const uint32 numElements;

                /**
                 * @param row           An object of template type `ValueRow` that provides access to all dense elements
                 *                      explicitly stored in the row
                 * @param indexIterator An iterator that provides access to the column indices of all dense elements
                 *                      explicitly stored in the row
                 */
                ConstRow(ValueRow row, IndexIterator indexIterator)
                    : row_(row), indexIterator_(indexIterator), numElements(row.size()) {}

                /**
                 * The type of the values in the row.
                 */
                typedef typename SparseSetView::value_type value_type;

                /**
                 * An iterator that provides read-only access to the values in the row.
                 */
                typedef typename SparseSetView::value_const_iterator const_iterator;

                /**
                 * Returns a `const_iterator` to the beginning of the row.
                 *
                 * @return A `const_iterator` to the beginning
                 */
                const_iterator cbegin() const {
                    return row_.cbegin();
                }

                /**
                 * Returns a `const_iterator` to the end of the row.
                 *
                 * @return A `const_iterator` to the end
                 */
                const_iterator cend() const {
                    return row_.cend();
                }

                /**
                 * Returns a pointer to the element that corresponds to a specific index.
                 *
                 * @param index The index of the element to be returned
                 * @return      A pointer to the element that corresponds to the given index or a null pointer, if no
                 *              such element is available
                 */
                const value_type* operator[](uint32 index) const {
                    uint32 i = indexIterator_[index];
                    return i == MAX_INDEX ? nullptr : &row_[i];
                }
        };

        /**
         * Provides read and write access to a single row of a `SparseSetView`.
         *
         * @tparam ValueRow         The type of the object that provides access to the values of all dense elements
         *                          explicitly stored in the row
         * @tparam IndexIterator    The type of the iterator that provides access to the column indices of all dense
         *                          elements explicitly stored in the row
         */
        template<typename ValueRow, typename IndexIterator>
        class Row final : public ConstRow<ValueRow, IndexIterator> {
            public:

                /**
                 * @param row           An object of template type `ValueRow` that provides access to the values of all
                 *                      dense elements explicitly stored in the row
                 * @param indexIterator An iterator that provides access to the column indices of all dense elements
                 *                      explicitly stored in the row
                 */
                Row(ValueRow row, IndexIterator indexIterator)
                    : ConstRow<ValueRow, IndexIterator>(row, indexIterator) {}

                /**
                 * An iterator that provides access to the values in the row and allows to modify them.
                 */
                typedef typename SparseSetView::value_iterator iterator;

                /**
                 * Returns an `iterator` to the beginning of the row.
                 *
                 * @return An `iterator` to the beginning
                 */
                iterator begin() {
                    return this->row_.begin();
                }

                /**
                 * Returns an `iterator` to the end of the row.
                 *
                 * @return An `iterator` to the end
                 */
                iterator end() {
                    return this->row_.end();
                }

                /**
                 * Returns a pointer to the element that corresponds to a specific index.
                 *
                 * @param index The index of the element to be returned
                 * @return      A pointer to the element that corresponds to the given index or a null pointer, if no
                 *              such element is available
                 */
                typename SparseSetView::value_type* operator[](uint32 index) {
                    uint32 i = this->indexIterator_[index];
                    return i == MAX_INDEX ? nullptr : &this->row_[i];
                }

                /**
                 * Returns a reference to the element that corresponds to a specific index. If no such element is
                 * available, it is inserted into the vector.
                 *
                 * @param index The index of the element to be returned
                 * @return      A reference to the element that corresponds to the given index
                 */
                typename SparseSetView::value_type& emplace(uint32 index) {
                    uint32 i = this->indexIterator_[index];

                    if (i == MAX_INDEX) {
                        this->indexIterator_[index] = static_cast<uint32>(this->row_.size());
                        this->row_.emplace_back(index);
                        return this->row_.back();
                    }

                    return this->row_[i];
                }

                /**
                 * Returns a reference to the element that corresponds to a specific index. If no such element is
                 * available, it is inserted into the vector using a specific default value.
                 *
                 * @param index         The index of the element to be returned
                 * @param defaultValue  The default value to be used
                 * @return              A reference to the element that corresponds to the given index
                 */
                typename SparseSetView::value_type& emplace(uint32 index, const T& defaultValue) {
                    uint32 i = this->indexIterator_[index];

                    if (i == MAX_INDEX) {
                        this->indexIterator_[index] = static_cast<uint32>(this->row_.size());
                        this->row_.emplace_back(index, defaultValue);
                        return this->row_.back();
                    }

                    return this->row_[i];
                }

                /**
                 * Removes the element that corresponds to a specific index, if available.
                 *
                 * @param index The index of the element to be removed
                 */
                void erase(uint32 index) {
                    uint32 i = this->indexIterator_[index];

                    if (i != MAX_INDEX) {
                        const value_type& lastEntry = this->row_.back();
                        uint32 lastIndex = lastEntry.index;

                        if (lastIndex != index) {
                            this->row_[i] = lastEntry;
                            this->indexIterator_[lastIndex] = i;
                        }

                        this->indexIterator_[index] = MAX_INDEX;
                        this->row_.pop_back();
                    }
                }

                /**
                 * Removes all elements from the row.
                 */
                void clear() {
                    while (!this->row_.empty()) {
                        const value_type& lastEntry = this->row_.back();
                        this->indexIterator_[lastEntry.index] = MAX_INDEX;
                        this->row_.pop_back();
                    }
                }
        };

    protected:

        /**
         * The index that is used to indicate that the value at a specific row and column is zero.
         */
        static inline constexpr uint32 MAX_INDEX = std::numeric_limits<uint32>::max();

    public:

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
         * Provides read-only access to an individual row in the view.
         */
        typedef ConstRow<typename AllocatedListOfLists<IndexedValue<T>>::const_row,
                         typename AllocatedCContiguousView<uint32>::value_const_iterator>
          const_row;

        /**
         * Provides access to an individual row in the view and allows to modify it.
         */
        typedef Row<typename AllocatedListOfLists<IndexedValue<T>>::row,
                    typename AllocatedCContiguousView<uint32>::value_iterator>
          row;

        /**
         * Creates and returns a view that provides read-only access to a specific row in the view.
         *
         * @param row   The index of the row
         * @return      A `const_row`
         */
        const_row operator[](uint32 row) const {
            return SparseSetView<T>::const_row(this->firstView[row], this->secondView.values_cbegin(row));
        }

        /**
         * Creates and returns a view that provides access to a specific row in the view and allows to modify it.
         *
         * @param row   The index of the row
         * @return      A `row`
         */
        row operator[](uint32 row) {
            return SparseSetView<T>::row(this->firstView[row], this->secondView.values_begin(row));
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
