#include "mlrl/seco/data/view_statistic_decomposable_sparse.hpp"

namespace seco {

    SparseDecomposableStatisticView::SparseDecomposableStatisticView(uint32 numRows, uint32 numCols)
        : Matrix(numRows, numCols), correctIndices_(numRows, numCols), incorrectIndices_(numRows, numCols) {}

    SparseDecomposableStatisticView::SparseDecomposableStatisticView(SparseDecomposableStatisticView&& other)
        : Matrix(other), correctIndices_(std::move(other.correctIndices_)),
          incorrectIndices_(std::move(other.incorrectIndices_)) {}

    typename SparseDecomposableStatisticView::index_const_iterator
      SparseDecomposableStatisticView::correct_indices_cbegin(uint32 row) const {
        return correctIndices_.values_cbegin(row);
    }

    typename SparseDecomposableStatisticView::index_const_iterator
      SparseDecomposableStatisticView::correct_indices_cend(uint32 row) const {
        return correctIndices_.values_cend(row);
    }

    typename SparseDecomposableStatisticView::index_iterator SparseDecomposableStatisticView::correct_indices_begin(
      uint32 row) {
        return correctIndices_.values_begin(row);
    }

    typename SparseDecomposableStatisticView::index_iterator SparseDecomposableStatisticView::correct_indices_end(
      uint32 row) {
        return correctIndices_.values_end(row);
    }

    typename SparseDecomposableStatisticView::const_row SparseDecomposableStatisticView::correct_indices_const_row(
      uint32 row) const {
        return correctIndices_[row];
    }

    typename SparseDecomposableStatisticView::row SparseDecomposableStatisticView::correct_indices_row(uint32 row) {
        return correctIndices_[row];
    }

    typename SparseDecomposableStatisticView::index_const_iterator
      SparseDecomposableStatisticView::incorrect_indices_cbegin(uint32 row) const {
        return incorrectIndices_.values_cbegin(row);
    }

    typename SparseDecomposableStatisticView::index_const_iterator
      SparseDecomposableStatisticView::incorrect_indices_cend(uint32 row) const {
        return incorrectIndices_.values_cend(row);
    }

    typename SparseDecomposableStatisticView::index_iterator SparseDecomposableStatisticView::incorrect_indices_begin(
      uint32 row) {
        return incorrectIndices_.values_begin(row);
    }

    typename SparseDecomposableStatisticView::index_iterator SparseDecomposableStatisticView::incorrect_indices_end(
      uint32 row) {
        return incorrectIndices_.values_end(row);
    }

    typename SparseDecomposableStatisticView::const_row SparseDecomposableStatisticView::incorrect_indices_const_row(
      uint32 row) const {
        return incorrectIndices_[row];
    }

    typename SparseDecomposableStatisticView::row SparseDecomposableStatisticView::incorrect_indices_row(uint32 row) {
        return incorrectIndices_[row];
    }

    void SparseDecomposableStatisticView::clear() {
        correctIndices_.clear();
        incorrectIndices_.clear();
    }
}
