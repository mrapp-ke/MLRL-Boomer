/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/view_matrix_lil.hpp"
#include "mlrl/seco/util/dll_exports.hpp"

namespace seco {

    /**
     * Implements row-wise read and write access to the indices of labels for which a rule predicts correctly or
     * incorrectly, respectively. The label indices are stored in two sparse matrices in the list of lists (LIL) format.
     */
    class MLRLSECO_API SparseDecomposableStatisticView : public Matrix {
        protected:

            /**
             * A LIL matrix that stores for each example the indices of labels for which a rule predicts correctly.
             */
            ListOfLists<uint32> correctIndices_;

            /**
             * A LIL matrix that stores for each example the indices of labels for which a rule predicts incorrectly.
             */
            ListOfLists<uint32> incorrectIndices_;

        public:

            /**
             * @param correctIndices    A pointer to an array of type `ListOfLists<uint32>::Row`, shape `(numRows)`,
             *                          that stores for each training example the indices of the labels for which a rule
             *                          predicts correctly
             * @param incorrectIndices  A pointer to an array of type `ListOfLists<uint32>::Row`, shape `(numRows)`,
             *                          that stores for each training example the indices of the labels for which a rule
             *                          predicts incorrectly
             * @param numRows           The number of rows in the view
             * @param numCols           The number of columns in the view
             */
            SparseDecomposableStatisticView(ListOfLists<uint32>::Row* correctIndices,
                                            ListOfLists<uint32>::Row* incorrectIndices, uint32 numRows, uint32 numCols)
                : Matrix(numRows, numCols), correctIndices_(correctIndices, numRows, numCols),
                  incorrectIndices_(incorrectIndices, numRows, numCols) {}

            /**
             * @param other A reference to an object of type `SparseDecomposableStatisticView` that should be copied
             */
            SparseDecomposableStatisticView(SparseDecomposableStatisticView&& other)
                : Matrix(other), correctIndices_(std::move(other.correctIndices_)),
                  incorrectIndices_(std::move(other.incorrectIndices_)) {}

            virtual ~SparseDecomposableStatisticView() override {}

            /**
             * Provides read-only access to an individual row in the view.
             */
            using const_row = typename ListOfLists<uint32>::const_row;

            /**
             * Provides access to an individual row in the view and allows to modify it.
             */
            using row = typename ListOfLists<uint32>::row;

            /**
             * An iterator that provides access to the label indices in the view and allows to modify them.
             */
            using index_iterator = typename ListOfLists<uint32>::value_iterator;

            /**
             * An iterator that provides read-only access to the label indices in the view.
             */
            using index_const_iterator = typename ListOfLists<uint32>::value_const_iterator;

            /**
             * Returns an `index_const_iterator` to the beginning of the indices at a specific row that correspond to
             * the labels for which a rule predicts correctly.
             *
             * @param row   The row
             * @return      An `index_const_iterator` to the beginning of the given row
             */
            index_const_iterator correct_indices_cbegin(uint32 row) const {
                return correctIndices_.values_cbegin(row);
            }

            /**
             * Returns an `index_const_iterator` to the end of the indices at a specific row that correspond to the
             * labels for which a rule predicts correctly.
             *
             * @param row   The row
             * @return      An `index_const_iterator` to the end of the given row
             */
            index_const_iterator correct_indices_cend(uint32 row) const {
                return correctIndices_.values_cend(row);
            }

            /**
             * Returns an `index_iterator` to the beginning of the indices at a specific row that correspond to the
             * labels for which a rule predicts correctly.
             *
             * @param row   The row
             * @return      An `index_iterator` to the beginning of the given row
             */
            index_iterator correct_indices_begin(uint32 row) {
                return correctIndices_.values_begin(row);
            }

            /**
             * Returns an `index_iterator` to the end of the indices at a specific row that correspond to the labels for
             * which a rule predicts correctly.
             *
             * @param row   The row
             * @return      An `index_iterator` to the end of the given row
             */
            index_iterator correct_indices_end(uint32 row) {
                return correctIndices_.values_end(row);
            }

            /**
             * Returns a view that provides read-only access to a specific row in the view that correspond to the labels
             * for which a rule predicts correctly.
             *
             * @param row   The index of the row
             * @return      A `const_row`
             */
            const_row correct_indices_row(uint32 row) const {
                return correctIndices_[row];
            }

            /**
             * Returns a view that provides access to a specific row in the view that correspond to the labels for which
             * a rule predicts correctly and allows to modify it.
             *
             * @param row   The index of the row
             * @return      A `row`
             */
            row correct_indices_row(uint32 row) {
                return correctIndices_[row];
            }

            /**
             * Returns an `index_const_iterator` to the beginning of the indices at a specific row that correspond to
             * the labels for which a rule predicts incorrectly.
             *
             * @param row   The row
             * @return      An `index_const_iterator` to the beginning of the given row
             */
            index_const_iterator incorrect_indices_cbegin(uint32 row) const {
                return incorrectIndices_.values_cbegin(row);
            }

            /**
             * Returns an `index_const_iterator` to the end of the indices at a specific row that correspond to the
             * labels for which a rule predicts incorrectly.
             *
             * @param row   The row
             * @return      An `index_const_iterator` to the end of the given row
             */
            index_const_iterator incorrect_indices_cend(uint32 row) const {
                return incorrectIndices_.values_cend(row);
            }

            /**
             * Returns an `index_iterator` to the beginning of the indices at a specific row that correspond to the
             * labels for which a rule predicts incorrectly.
             *
             * @param row   The row
             * @return      An `index_iterator` to the beginning of the given row
             */
            index_iterator incorrect_indices_begin(uint32 row) {
                return incorrectIndices_.values_begin(row);
            }

            /**
             * Returns an `index_iterator` to the end of the indices at a specific row that correspond to the labels for
             * which a rule predicts incorrectly.
             *
             * @param row   The row
             * @return      An `index_iterator` to the end of the given row
             */
            index_iterator incorrect_indices_end(uint32 row) {
                return incorrectIndices_.values_end(row);
            }

            /**
             * Returns a view that provides read-only access to a specific row in the view that correspond to the labels
             * for which a rule predicts incorrectly.
             *
             * @param row   The index of the row
             * @return      A `const_row`
             */
            const_row incorrect_indices_row(uint32 row) const {
                return incorrectIndices_[row];
            }

            /**
             * Returns a view that provides access to a specific row in the view that correspond to the labels for which
             * a rule predicts incorrectly and allows to modify it.
             *
             * @param row   The index of the row
             * @return      A `row`
             */
            row incorrect_indices_row(uint32 row) {
                return incorrectIndices_[row];
            }

            /**
             * Sets all values stored in the view to zero.
             */
            void clear() {
                correctIndices_.clear();
                incorrectIndices_.clear();
            }
    };

    /**
     * Allocates the memory, a `SparseDecomposableStatisticView` provides access to.
     *
     * @tparam View             The type of the view
     * @tparam MemoryAllocator  The type of the memory allocator to be used
     */
    template<typename View, typename MemoryAllocator = DefaultMemoryAllocator>
    class MLRLSECO_API SparseDecomposableStatisticViewAllocator : public View {
        private:

            AllocatedListOfLists<uint32, MemoryAllocator> correctIndicesAllocator_;

            AllocatedListOfLists<uint32, MemoryAllocator> incorrectIndicesAllocator_;

        public:

            /**
             * @param numRows   The number of rows in the view
             * @param numCols   The number of columns in the view
             */
            explicit SparseDecomposableStatisticViewAllocator(uint32 numRows, uint32 numCols)
                : View(nullptr, nullptr, numRows, numCols), correctIndicesAllocator_(numRows, numCols),
                  incorrectIndicesAllocator_(numRows, numCols) {
                View::correctIndices_.array = correctIndicesAllocator_.array;
                View::incorrectIndices_.array = incorrectIndicesAllocator_.array;
            }

            /**
             * @param other A reference to an object of type `SparseDecomposableStatisticViewAllocator` that should be
             * copied
             */
            SparseDecomposableStatisticViewAllocator(
              const SparseDecomposableStatisticViewAllocator<View, MemoryAllocator>& other) = delete;

            /**
             * @param other A reference to an object of type `SparseDecomposableStatisticViewAllocator` that should be
             * moved
             */
            SparseDecomposableStatisticViewAllocator(
              SparseDecomposableStatisticViewAllocator<View, MemoryAllocator>&& other)
                : View(std::move(other)), correctIndicesAllocator_(std::move(other.correctIndicesAllocator_)),
                  incorrectIndicesAllocator_(std::move(other.incorrectIndicesAllocator_)) {}

            virtual ~SparseDecomposableStatisticViewAllocator() override {}

            /**
             * Provides read-only access to an individual row in the view.
             */
            using const_row = AllocatedListOfLists<uint32, MemoryAllocator>::const_row;

            /**
             * Provides access to an individual row in the view and allows to modify it.
             */
            using row = AllocatedListOfLists<uint32, MemoryAllocator>::row;

            /**
             * Returns a view that provides read-only access to a specific row in the view that correspond to the labels
             * for which a rule predicts correctly.
             *
             * @param row   The index of the row
             * @return      A `const_row`
             */
            const_row correct_indices_row(uint32 row) const {
                return correctIndicesAllocator_[row];
            }

            /**
             * Returns a view that provides access to a specific row in the view that correspond to the labels for which
             * a rule predicts correctly and allows to modify it.
             *
             * @param row   The index of the row
             * @return      A `row`
             */
            row correct_indices_row(uint32 row) {
                return correctIndicesAllocator_[row];
            }

            /**
             * Returns a view that provides read-only access to a specific row in the view that correspond to the labels
             * for which a rule predicts incorrectly.
             *
             * @param row   The index of the row
             * @return      A `const_row`
             */
            const_row incorrect_indices_row(uint32 row) const {
                return incorrectIndicesAllocator_[row];
            }

            /**
             * Returns a view that provides access to a specific row in the view that correspond to the labels for which
             * a rule predicts incorrectly and allows to modify it.
             *
             * @param row   The index of the row
             * @return      A `row`
             */
            row incorrect_indices_row(uint32 row) {
                return incorrectIndicesAllocator_[row];
            }
    };

    /**
     * Allocates the memory, a `SparseDecomposableStatisticView` provides access to.
     *
     * @tparam MemoryAllocator The type of the memory allocator to be used
     */
    template<typename MemoryAllocator = DefaultMemoryAllocator>
    using AllocatedSparseDecomposableStatisticView =
      SparseDecomposableStatisticViewAllocator<SparseDecomposableStatisticView, MemoryAllocator>;
}
