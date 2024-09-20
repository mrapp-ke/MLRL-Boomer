#include "mlrl/boosting/data/view_statistic_non_decomposable_dense.hpp"

#include "mlrl/boosting/util/math.hpp"

namespace boosting {

    DenseNonDecomposableStatisticView::DenseNonDecomposableStatisticView(uint32 numRows, uint32 numCols)
        : CompositeMatrix<AllocatedCContiguousView<float64>, AllocatedCContiguousView<float64>>(
            AllocatedCContiguousView<float64>(numRows, numCols),
            AllocatedCContiguousView<float64>(numRows, util::triangularNumber(numCols)), numRows, numCols) {}

    DenseNonDecomposableStatisticView::DenseNonDecomposableStatisticView(DenseNonDecomposableStatisticView&& other)
        : CompositeMatrix<AllocatedCContiguousView<float64>, AllocatedCContiguousView<float64>>(std::move(other)) {}

    DenseNonDecomposableStatisticView::gradient_const_iterator DenseNonDecomposableStatisticView::gradients_cbegin(
      uint32 row) const {
        return CompositeMatrix::firstView.values_cbegin(row);
    }

    DenseNonDecomposableStatisticView::gradient_const_iterator DenseNonDecomposableStatisticView::gradients_cend(
      uint32 row) const {
        return CompositeMatrix::firstView.values_cend(row);
    }

    DenseNonDecomposableStatisticView::gradient_iterator DenseNonDecomposableStatisticView::gradients_begin(
      uint32 row) {
        return CompositeMatrix::firstView.values_begin(row);
    }

    DenseNonDecomposableStatisticView::gradient_iterator DenseNonDecomposableStatisticView::gradients_end(uint32 row) {
        return CompositeMatrix::firstView.values_end(row);
    }

    DenseNonDecomposableStatisticView::hessian_const_iterator DenseNonDecomposableStatisticView::hessians_cbegin(
      uint32 row) const {
        return CompositeMatrix::secondView.values_cbegin(row);
    }

    DenseNonDecomposableStatisticView::hessian_const_iterator DenseNonDecomposableStatisticView::hessians_cend(
      uint32 row) const {
        return CompositeMatrix::secondView.values_cend(row);
    }

    DenseNonDecomposableStatisticView::hessian_iterator DenseNonDecomposableStatisticView::hessians_begin(uint32 row) {
        return CompositeMatrix::secondView.values_begin(row);
    }

    DenseNonDecomposableStatisticView::hessian_iterator DenseNonDecomposableStatisticView::hessians_end(uint32 row) {
        return CompositeMatrix::secondView.values_end(row);
    }

    DenseNonDecomposableStatisticView::hessian_diagonal_const_iterator
      DenseNonDecomposableStatisticView::hessians_diagonal_cbegin(uint32 row) const {
        return hessian_diagonal_const_iterator(CompositeMatrix::secondView[row], 0);
    }

    DenseNonDecomposableStatisticView::hessian_diagonal_const_iterator
      DenseNonDecomposableStatisticView::hessians_diagonal_cend(uint32 row) const {
        return hessian_diagonal_const_iterator(CompositeMatrix::secondView[row], Matrix::numCols);
    }
}
