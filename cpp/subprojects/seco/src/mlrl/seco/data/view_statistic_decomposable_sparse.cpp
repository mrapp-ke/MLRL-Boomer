#include "mlrl/seco/data/view_statistic_decomposable_sparse.hpp"

namespace seco {

    SparseDecomposableStatisticView::SparseDecomposableStatisticView(uint32 numRows, uint32 numCols)
        : CompositeMatrix<AllocatedListOfLists<uint32>, AllocatedListOfLists<uint32>>(
            AllocatedListOfLists<uint32>(numRows, numCols), AllocatedListOfLists<uint32>(numRows, numCols), numRows,
            numCols) {}

    SparseDecomposableStatisticView::SparseDecomposableStatisticView(SparseDecomposableStatisticView&& other)
        : CompositeMatrix<AllocatedListOfLists<uint32>, AllocatedListOfLists<uint32>>(std::move(other)) {}

    typename SparseDecomposableStatisticView::index_const_iterator
      SparseDecomposableStatisticView::correct_indices_cbegin(uint32 row) const {
        return this->firstView.values_cbegin(row);
    }

    typename SparseDecomposableStatisticView::index_const_iterator
      SparseDecomposableStatisticView::correct_indices_cend(uint32 row) const {
        return this->firstView.values_cend(row);
    }

    typename SparseDecomposableStatisticView::index_iterator SparseDecomposableStatisticView::correct_indices_begin(
      uint32 row) {
        return this->firstView.values_begin(row);
    }

    typename SparseDecomposableStatisticView::index_iterator SparseDecomposableStatisticView::correct_indices_end(
      uint32 row) {
        return this->firstView.values_end(row);
    }

    typename SparseDecomposableStatisticView::const_row SparseDecomposableStatisticView::correct_indices_const_row(
      uint32 row) const {
        return this->firstView[row];
    }

    typename SparseDecomposableStatisticView::row SparseDecomposableStatisticView::correct_indices_row(uint32 row) {
        return this->firstView[row];
    }

    typename SparseDecomposableStatisticView::index_const_iterator
      SparseDecomposableStatisticView::incorrect_indices_cbegin(uint32 row) const {
        return this->secondView.values_cbegin(row);
    }

    typename SparseDecomposableStatisticView::index_const_iterator
      SparseDecomposableStatisticView::incorrect_indices_cend(uint32 row) const {
        return this->secondView.values_cend(row);
    }

    typename SparseDecomposableStatisticView::index_iterator SparseDecomposableStatisticView::incorrect_indices_begin(
      uint32 row) {
        return this->secondView.values_begin(row);
    }

    typename SparseDecomposableStatisticView::index_iterator SparseDecomposableStatisticView::incorrect_indices_end(
      uint32 row) {
        return this->secondView.values_end(row);
    }

    typename SparseDecomposableStatisticView::const_row SparseDecomposableStatisticView::incorrect_indices_const_row(
      uint32 row) const {
        return this->secondView[row];
    }

    typename SparseDecomposableStatisticView::row SparseDecomposableStatisticView::incorrect_indices_row(uint32 row) {
        return this->secondView[row];
    }
}
