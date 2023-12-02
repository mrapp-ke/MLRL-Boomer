#include "mlrl/boosting/data/statistic_view_label_wise_dense.hpp"

#include "mlrl/common/util/view_functions.hpp"

namespace boosting {

    DenseLabelWiseStatisticView::DenseLabelWiseStatisticView(uint32 numRows, uint32 numCols, Tuple<float64>* statistics)
        : numRows_(numRows), numCols_(numCols), statistics_(statistics) {}

    DenseLabelWiseStatisticView::const_iterator DenseLabelWiseStatisticView::cbegin(uint32 row) const {
        return &statistics_[row * numCols_];
    }

    DenseLabelWiseStatisticView::const_iterator DenseLabelWiseStatisticView::cend(uint32 row) const {
        return &statistics_[(row + 1) * numCols_];
    }

    DenseLabelWiseStatisticView::iterator DenseLabelWiseStatisticView::begin(uint32 row) {
        return &statistics_[row * numCols_];
    }

    DenseLabelWiseStatisticView::iterator DenseLabelWiseStatisticView::end(uint32 row) {
        return &statistics_[(row + 1) * numCols_];
    }

    void DenseLabelWiseStatisticView::clear() {
        setViewToZeros(statistics_, numRows_ * numCols_);
    }

    void DenseLabelWiseStatisticView::addToRow(uint32 row, const_iterator begin, const_iterator end, float64 weight) {
        uint32 offset = row * numCols_;
        addToView(&statistics_[offset], begin, numCols_, weight);
    }

    uint32 DenseLabelWiseStatisticView::getNumRows() const {
        return numRows_;
    }

    uint32 DenseLabelWiseStatisticView::getNumCols() const {
        return numCols_;
    }
}
