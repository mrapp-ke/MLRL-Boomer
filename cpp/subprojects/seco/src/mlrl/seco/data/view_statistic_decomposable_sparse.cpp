#include "mlrl/seco/data/view_statistic_decomposable_sparse.hpp"

namespace seco {

    SparseDecomposableStatisticView::SparseDecomposableStatisticView(uint32 numRows, uint32 numCols)
        : CompositeMatrix<CompositeMatrix<AllocatedListOfLists<uint32>, AllocatedListOfLists<uint32>>,
                          CompositeMatrix<AllocatedListOfLists<uint32>, AllocatedListOfLists<uint32>>>(
            CompositeMatrix<AllocatedListOfLists<uint32>, AllocatedListOfLists<uint32>>(
              AllocatedListOfLists<uint32>(numRows, numCols), AllocatedListOfLists<uint32>(numRows, numCols), numRows,
              numCols),
            CompositeMatrix<AllocatedListOfLists<uint32>, AllocatedListOfLists<uint32>>(
              AllocatedListOfLists<uint32>(numRows, numCols), AllocatedListOfLists<uint32>(numRows, numCols), numRows,
              numCols),
            numRows, numCols) {}

    SparseDecomposableStatisticView::SparseDecomposableStatisticView(SparseDecomposableStatisticView&& other)
        : CompositeMatrix<CompositeMatrix<AllocatedListOfLists<uint32>, AllocatedListOfLists<uint32>>,
                          CompositeMatrix<AllocatedListOfLists<uint32>, AllocatedListOfLists<uint32>>>(
            std::move(other)) {}

    typename SparseDecomposableStatisticView::index_const_iterator SparseDecomposableStatisticView::in_indices_cbegin(
      uint32 row) const {
        return this->firstView.firstView.values_cbegin(row);
    }

    typename SparseDecomposableStatisticView::index_const_iterator SparseDecomposableStatisticView::in_indices_cend(
      uint32 row) const {
        return this->firstView.firstView.values_cend(row);
    }

    typename SparseDecomposableStatisticView::index_iterator SparseDecomposableStatisticView::in_indices_begin(
      uint32 row) {
        return this->firstView.firstView.values_begin(row);
    }

    typename SparseDecomposableStatisticView::index_iterator SparseDecomposableStatisticView::in_indices_end(
      uint32 row) {
        return this->firstView.firstView.values_end(row);
    }

    typename SparseDecomposableStatisticView::const_row SparseDecomposableStatisticView::in_const_row(
      uint32 row) const {
        return this->firstView.firstView[row];
    }

    typename SparseDecomposableStatisticView::row SparseDecomposableStatisticView::in_row(uint32 row) {
        return this->firstView.firstView[row];
    }

    typename SparseDecomposableStatisticView::index_const_iterator SparseDecomposableStatisticView::ip_indices_cbegin(
      uint32 row) const {
        return this->firstView.secondView.values_cbegin(row);
    }

    typename SparseDecomposableStatisticView::index_const_iterator SparseDecomposableStatisticView::ip_indices_cend(
      uint32 row) const {
        return this->firstView.secondView.values_cend(row);
    }

    typename SparseDecomposableStatisticView::index_iterator SparseDecomposableStatisticView::ip_indices_begin(
      uint32 row) {
        return this->firstView.secondView.values_begin(row);
    }

    typename SparseDecomposableStatisticView::index_iterator SparseDecomposableStatisticView::ip_indices_end(
      uint32 row) {
        return this->firstView.secondView.values_end(row);
    }

    typename SparseDecomposableStatisticView::const_row SparseDecomposableStatisticView::ip_const_row(
      uint32 row) const {
        return this->firstView.secondView[row];
    }

    typename SparseDecomposableStatisticView::row SparseDecomposableStatisticView::ip_row(uint32 row) {
        return this->firstView.secondView[row];
    }

    typename SparseDecomposableStatisticView::index_const_iterator SparseDecomposableStatisticView::rn_indices_cbegin(
      uint32 row) const {
        return this->secondView.firstView.values_cbegin(row);
    }

    typename SparseDecomposableStatisticView::index_const_iterator SparseDecomposableStatisticView::rn_indices_cend(
      uint32 row) const {
        return this->secondView.firstView.values_cend(row);
    }

    typename SparseDecomposableStatisticView::index_iterator SparseDecomposableStatisticView::rn_indices_begin(
      uint32 row) {
        return this->secondView.firstView.values_begin(row);
    }

    typename SparseDecomposableStatisticView::index_iterator SparseDecomposableStatisticView::rn_indices_end(
      uint32 row) {
        return this->secondView.firstView.values_end(row);
    }

    typename SparseDecomposableStatisticView::const_row SparseDecomposableStatisticView::rn_const_row(
      uint32 row) const {
        return this->secondView.firstView[row];
    }

    typename SparseDecomposableStatisticView::row SparseDecomposableStatisticView::rn_row(uint32 row) {
        return this->secondView.firstView[row];
    }

    typename SparseDecomposableStatisticView::index_const_iterator SparseDecomposableStatisticView::rp_indices_cbegin(
      uint32 row) const {
        return this->secondView.secondView.values_cbegin(row);
    }

    typename SparseDecomposableStatisticView::index_const_iterator SparseDecomposableStatisticView::rp_indices_cend(
      uint32 row) const {
        return this->secondView.secondView.values_cend(row);
    }

    typename SparseDecomposableStatisticView::index_iterator SparseDecomposableStatisticView::rp_indices_begin(
      uint32 row) {
        return this->secondView.secondView.values_begin(row);
    }

    typename SparseDecomposableStatisticView::index_iterator SparseDecomposableStatisticView::rp_indices_end(
      uint32 row) {
        return this->secondView.secondView.values_end(row);
    }

    typename SparseDecomposableStatisticView::const_row SparseDecomposableStatisticView::rp_const_row(
      uint32 row) const {
        return this->secondView.secondView[row];
    }

    typename SparseDecomposableStatisticView::row SparseDecomposableStatisticView::rp_row(uint32 row) {
        return this->secondView.secondView[row];
    }
}
