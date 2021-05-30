#include "boosting/data/statistic_view_dense_label_wise.hpp"
#include "boosting/data/arrays.hpp"
#include "common/data/arrays.hpp"


namespace boosting {

    DenseLabelWiseStatisticConstView::DenseLabelWiseStatisticConstView(uint32 numRows, uint32 numCols,
                                                                       float64* gradients, float64* hessians)
        : numRows_(numRows), numCols_(numCols), gradients_(gradients), hessians_(hessians) {

    }

    DenseLabelWiseStatisticConstView::gradient_const_iterator DenseLabelWiseStatisticConstView::gradients_row_cbegin(
            uint32 row) const {
        return &gradients_[row * numCols_];
    }

    DenseLabelWiseStatisticConstView::gradient_const_iterator DenseLabelWiseStatisticConstView::gradients_row_cend(
            uint32 row) const {
        return &gradients_[(row + 1) * numCols_];
    }

    DenseLabelWiseStatisticConstView::hessian_const_iterator DenseLabelWiseStatisticConstView::hessians_row_cbegin(
            uint32 row) const {
        return &hessians_[row * numCols_];
    }

    DenseLabelWiseStatisticConstView::hessian_const_iterator DenseLabelWiseStatisticConstView::hessians_row_cend(
            uint32 row) const {
        return &hessians_[(row + 1) * numCols_];
    }

    uint32 DenseLabelWiseStatisticConstView::getNumRows() const {
        return numRows_;
    }

    uint32 DenseLabelWiseStatisticConstView::getNumCols() const {
        return numCols_;
    }

    DenseLabelWiseStatisticView::DenseLabelWiseStatisticView(uint32 numRows, uint32 numCols, float64* gradients,
                                                             float64* hessians)
        : DenseLabelWiseStatisticConstView(numRows, numCols, gradients, hessians) {

    }

    DenseLabelWiseStatisticView::gradient_iterator DenseLabelWiseStatisticView::gradients_row_begin(uint32 row) {
        return &gradients_[row * numCols_];
    }

    DenseLabelWiseStatisticView::gradient_iterator DenseLabelWiseStatisticView::gradients_row_end(uint32 row) {
        return &gradients_[(row + 1) * numCols_];
    }

    DenseLabelWiseStatisticView::hessian_iterator DenseLabelWiseStatisticView::hessians_row_begin(uint32 row) {
        return &hessians_[row * numCols_];
    }

    DenseLabelWiseStatisticView::hessian_iterator DenseLabelWiseStatisticView::hessians_row_end(uint32 row) {
        return &hessians_[(row + 1) * numCols_];
    }

    void DenseLabelWiseStatisticView::clear() {
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
