#include "boosting/data/statistic_view_dense_example_wise.hpp"
#include "boosting/math/math.hpp"
#include "boosting/data/arrays.hpp"
#include "common/data/arrays.hpp"
#include <cstdlib>


namespace boosting {

    DenseExampleWiseStatisticView::DenseExampleWiseStatisticView(uint32 numRows, uint32 numGradients)
        : DenseExampleWiseStatisticView(numRows, numGradients, false) {

    }

    DenseExampleWiseStatisticView::DenseExampleWiseStatisticView(uint32 numRows, uint32 numGradients, bool init)
        : numRows_(numRows), numGradients_(numGradients), numHessians_(triangularNumber(numGradients)),
          gradients_((float64*) (init ? calloc(numRows * numGradients, sizeof(float64))
                                      : malloc(numRows * numGradients * sizeof(float64)))),
          hessians_((float64*) (init ? calloc(numRows * numHessians_, sizeof(float64))
                                     : malloc(numRows * numHessians_ * sizeof(float64)))) {

    }

    DenseExampleWiseStatisticView::~DenseExampleWiseStatisticView() {
        free(gradients_);
        free(hessians_);
    }

    DenseExampleWiseStatisticView::gradient_iterator DenseExampleWiseStatisticView::gradients_row_begin(
            uint32 row) {
        return &gradients_[row * numGradients_];
    }

    DenseExampleWiseStatisticView::gradient_iterator DenseExampleWiseStatisticView::gradients_row_end(uint32 row) {
        return &gradients_[(row + 1) * numGradients_];
    }

    DenseExampleWiseStatisticView::gradient_const_iterator DenseExampleWiseStatisticView::gradients_row_cbegin(
            uint32 row) const {
        return &gradients_[row * numGradients_];
    }

    DenseExampleWiseStatisticView::gradient_const_iterator DenseExampleWiseStatisticView::gradients_row_cend(
            uint32 row) const {
        return &gradients_[(row + 1) * numGradients_];
    }

    DenseExampleWiseStatisticView::hessian_iterator DenseExampleWiseStatisticView::hessians_row_begin(uint32 row) {
        return &hessians_[row * numHessians_];
    }

    DenseExampleWiseStatisticView::hessian_iterator DenseExampleWiseStatisticView::hessians_row_end(uint32 row) {
        return &hessians_[(row + 1) * numHessians_];
    }

    DenseExampleWiseStatisticView::hessian_const_iterator DenseExampleWiseStatisticView::hessians_row_cbegin(
            uint32 row) const {
        return &hessians_[row * numHessians_];
    }

    DenseExampleWiseStatisticView::hessian_const_iterator DenseExampleWiseStatisticView::hessians_row_cend(
            uint32 row) const {
        return &hessians_[(row + 1) * numHessians_];
    }

    uint32 DenseExampleWiseStatisticView::getNumRows() const {
        return numRows_;
    }

    uint32 DenseExampleWiseStatisticView::getNumCols() const {
        return numGradients_;
    }

    void DenseExampleWiseStatisticView::setAllToZero() {
        setArrayToZeros(gradients_, numRows_ * numGradients_);
        setArrayToZeros(hessians_, numRows_ * numHessians_);
    }

    void DenseExampleWiseStatisticView::addToRow(uint32 row, gradient_const_iterator gradientsBegin,
                                                   gradient_const_iterator gradientsEnd,
                                                   hessian_const_iterator hessiansBegin,
                                                   hessian_const_iterator hessiansEnd, float64 weight) {
        addToArray(&gradients_[row * numGradients_], gradientsBegin, numGradients_, weight);
        addToArray(&hessians_[row * numHessians_], hessiansBegin, numHessians_, weight);
    }

}
