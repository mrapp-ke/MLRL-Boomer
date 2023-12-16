#include "mlrl/boosting/data/view_statistic_example_wise_dense.hpp"

#include "mlrl/boosting/util/math.hpp"

namespace boosting {

    DenseExampleWiseStatisticView::DenseExampleWiseStatisticView(uint32 numRows, uint32 numCols)
        : CompositeMatrix<AllocatedCContiguousView<float64>, AllocatedCContiguousView<float64>>(
          AllocatedCContiguousView<float64>(numRows, numCols),
          AllocatedCContiguousView<float64>(numRows, triangularNumber(numCols)), numRows, numCols) {}

    DenseExampleWiseStatisticView::DenseExampleWiseStatisticView(DenseExampleWiseStatisticView&& other)
        : CompositeMatrix<AllocatedCContiguousView<float64>, AllocatedCContiguousView<float64>>(std::move(other)) {}

    DenseExampleWiseStatisticView::gradient_const_iterator DenseExampleWiseStatisticView::gradients_cbegin(
      uint32 row) const {
        return CompositeMatrix::firstView.values_cbegin(row);
    }

    DenseExampleWiseStatisticView::gradient_const_iterator DenseExampleWiseStatisticView::gradients_cend(
      uint32 row) const {
        return CompositeMatrix::firstView.values_cend(row);
    }

    DenseExampleWiseStatisticView::gradient_iterator DenseExampleWiseStatisticView::gradients_begin(uint32 row) {
        return CompositeMatrix::firstView.values_begin(row);
    }

    DenseExampleWiseStatisticView::gradient_iterator DenseExampleWiseStatisticView::gradients_end(uint32 row) {
        return CompositeMatrix::firstView.values_end(row);
    }

    DenseExampleWiseStatisticView::hessian_const_iterator DenseExampleWiseStatisticView::hessians_cbegin(
      uint32 row) const {
        return CompositeMatrix::secondView.values_cbegin(row);
    }

    DenseExampleWiseStatisticView::hessian_const_iterator DenseExampleWiseStatisticView::hessians_cend(
      uint32 row) const {
        return CompositeMatrix::secondView.values_cend(row);
    }

    DenseExampleWiseStatisticView::hessian_iterator DenseExampleWiseStatisticView::hessians_begin(uint32 row) {
        return CompositeMatrix::secondView.values_begin(row);
    }

    DenseExampleWiseStatisticView::hessian_iterator DenseExampleWiseStatisticView::hessians_end(uint32 row) {
        return CompositeMatrix::secondView.values_end(row);
    }

    DenseExampleWiseStatisticView::hessian_diagonal_const_iterator
      DenseExampleWiseStatisticView::hessians_diagonal_cbegin(uint32 row) const {
        return DiagonalConstIterator<float64>(CompositeMatrix::secondView.values_cbegin(row), 0);
    }

    DenseExampleWiseStatisticView::hessian_diagonal_const_iterator
      DenseExampleWiseStatisticView::hessians_diagonal_cend(uint32 row) const {
        return DiagonalConstIterator<float64>(CompositeMatrix::secondView.values_cbegin(row), Matrix::numCols);
    }
}
