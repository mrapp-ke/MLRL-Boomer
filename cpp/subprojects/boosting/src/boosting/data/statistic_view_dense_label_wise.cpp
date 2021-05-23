#include "boosting/data/statistic_view_dense_label_wise.hpp"
#include "boosting/data/arrays.hpp"
#include "common/data/arrays.hpp"
#include <cstdlib>


namespace boosting {

    DenseLabelWiseStatisticView::DenseLabelWiseStatisticView(uint32 numRows, uint32 numCols)
        : DenseLabelWiseStatisticView(numRows, numCols, false) {

    }

    DenseLabelWiseStatisticView::DenseLabelWiseStatisticView(uint32 numRows, uint32 numCols, bool init)
        : numRows_(numRows), numCols_(numCols),
          gradients_((float64*) (init ? calloc(numRows * numCols, sizeof(float64))
                                      : malloc(numRows * numCols * sizeof(float64)))),
          hessians_((float64*) (init ? calloc(numRows * numCols, sizeof(float64))
                                     : malloc(numRows * numCols * sizeof(float64)))) {

    }

    DenseLabelWiseStatisticView::~DenseLabelWiseStatisticView() {
        free(gradients_);
        free(hessians_);
    }

    DenseLabelWiseStatisticView::gradient_iterator DenseLabelWiseStatisticView::gradients_row_begin(uint32 row) {
        return &gradients_[row * numCols_];
    }

    DenseLabelWiseStatisticView::gradient_iterator DenseLabelWiseStatisticView::gradients_row_end(uint32 row) {
        return &gradients_[(row + 1) * numCols_];
    }

    DenseLabelWiseStatisticView::gradient_const_iterator DenseLabelWiseStatisticView::gradients_row_cbegin(
            uint32 row) const {
        return &gradients_[row * numCols_];
    }

    DenseLabelWiseStatisticView::gradient_const_iterator DenseLabelWiseStatisticView::gradients_row_cend(
            uint32 row) const {
        return &gradients_[(row + 1) * numCols_];
    }

    DenseLabelWiseStatisticView::hessian_iterator DenseLabelWiseStatisticView::hessians_row_begin(uint32 row) {
        return &hessians_[row * numCols_];
    }

    DenseLabelWiseStatisticView::hessian_iterator DenseLabelWiseStatisticView::hessians_row_end(uint32 row) {
        return &hessians_[(row + 1) * numCols_];
    }

    DenseLabelWiseStatisticView::hessian_const_iterator DenseLabelWiseStatisticView::hessians_row_cbegin(
            uint32 row) const {
        return &hessians_[row * numCols_];
    }

    DenseLabelWiseStatisticView::hessian_const_iterator DenseLabelWiseStatisticView::hessians_row_cend(
            uint32 row) const {
        return &hessians_[(row + 1) * numCols_];
    }

    uint32 DenseLabelWiseStatisticView::getNumRows() const {
        return numRows_;
    }

    uint32 DenseLabelWiseStatisticView::getNumCols() const {
        return numCols_;
    }

    void DenseLabelWiseStatisticView::setAllToZero() {
        setArrayToZeros(gradients_, numRows_ * numCols_);
        setArrayToZeros(hessians_, numRows_ * numCols_);
    }

    void DenseLabelWiseStatisticView::addToRow(uint32 row, gradient_const_iterator gradientsBegin,
                                                 gradient_const_iterator gradientsEnd,
                                                 hessian_const_iterator hessiansBegin,
                                                 hessian_const_iterator hessiansEnd, float64 weight) {
        uint32 offset = row * numCols_;
        addToArray(&gradients_[offset], gradientsBegin, numCols_, weight);
        addToArray(&hessians_[offset], hessiansBegin, numCols_, weight);
    }

}
