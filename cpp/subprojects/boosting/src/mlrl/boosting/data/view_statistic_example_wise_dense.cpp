#include "mlrl/boosting/data/view_statistic_example_wise_dense.hpp"

#include "mlrl/common/util/view_functions.hpp"

namespace boosting {

    DenseExampleWiseStatisticView::DenseExampleWiseStatisticView(uint32 numRows, uint32 numGradients,
                                                                 uint32 numHessians, float64* gradients,
                                                                 float64* hessians)
        : numRows_(numRows), numGradients_(numGradients), numHessians_(numHessians), gradients_(gradients),
          hessians_(hessians) {}

    DenseExampleWiseStatisticView::gradient_const_iterator DenseExampleWiseStatisticView::gradients_cbegin(
      uint32 row) const {
        return &gradients_[row * numGradients_];
    }

    DenseExampleWiseStatisticView::gradient_const_iterator DenseExampleWiseStatisticView::gradients_cend(
      uint32 row) const {
        return &gradients_[(row + 1) * numGradients_];
    }

    DenseExampleWiseStatisticView::gradient_iterator DenseExampleWiseStatisticView::gradients_begin(uint32 row) {
        return &gradients_[row * numGradients_];
    }

    DenseExampleWiseStatisticView::gradient_iterator DenseExampleWiseStatisticView::gradients_end(uint32 row) {
        return &gradients_[(row + 1) * numGradients_];
    }

    DenseExampleWiseStatisticView::hessian_const_iterator DenseExampleWiseStatisticView::hessians_cbegin(
      uint32 row) const {
        return &hessians_[row * numHessians_];
    }

    DenseExampleWiseStatisticView::hessian_const_iterator DenseExampleWiseStatisticView::hessians_cend(
      uint32 row) const {
        return &hessians_[(row + 1) * numHessians_];
    }

    DenseExampleWiseStatisticView::hessian_diagonal_const_iterator
      DenseExampleWiseStatisticView::hessians_diagonal_cbegin(uint32 row) const {
        return DiagonalConstIterator<float64>(&hessians_[row * numHessians_], 0);
    }

    DenseExampleWiseStatisticView::hessian_diagonal_const_iterator
      DenseExampleWiseStatisticView::hessians_diagonal_cend(uint32 row) const {
        return DiagonalConstIterator<float64>(&hessians_[row * numHessians_], numGradients_);
    }

    DenseExampleWiseStatisticView::hessian_iterator DenseExampleWiseStatisticView::hessians_begin(uint32 row) {
        return &hessians_[row * numHessians_];
    }

    DenseExampleWiseStatisticView::hessian_iterator DenseExampleWiseStatisticView::hessians_end(uint32 row) {
        return &hessians_[(row + 1) * numHessians_];
    }

    void DenseExampleWiseStatisticView::clear() {
        setViewToZeros(gradients_, numRows_ * numGradients_);
        setViewToZeros(hessians_, numRows_ * numHessians_);
    }

    void DenseExampleWiseStatisticView::addToRow(uint32 row, gradient_const_iterator gradientsBegin,
                                                 gradient_const_iterator gradientsEnd,
                                                 hessian_const_iterator hessiansBegin,
                                                 hessian_const_iterator hessiansEnd, float64 weight) {
        addToView(&gradients_[row * numGradients_], gradientsBegin, numGradients_, weight);
        addToView(&hessians_[row * numHessians_], hessiansBegin, numHessians_, weight);
    }

    uint32 DenseExampleWiseStatisticView::getNumRows() const {
        return numRows_;
    }

    uint32 DenseExampleWiseStatisticView::getNumCols() const {
        return numGradients_;
    }
}
