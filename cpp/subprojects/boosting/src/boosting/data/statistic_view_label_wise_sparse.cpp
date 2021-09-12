#include "boosting/data/statistic_view_label_wise_sparse.hpp"


namespace boosting {

    SparseLabelWiseStatisticConstView::SparseLabelWiseStatisticConstView(LilMatrix<Tuple<float64>>* statistics,
                                                                         uint32 numCols)
        : statistics_(statistics), numCols_(numCols) {

    }

    SparseLabelWiseStatisticConstView::const_iterator SparseLabelWiseStatisticConstView::row_cbegin(uint32 row) const {
        return statistics_->getRow(row).cbegin();
    }

    SparseLabelWiseStatisticConstView::const_iterator SparseLabelWiseStatisticConstView::row_cend(uint32 row) const {
        return statistics_->getRow(row).cend();
    }

    uint32 SparseLabelWiseStatisticConstView::getNumRows() const {
        return statistics_->getNumRows();
    }

    uint32 SparseLabelWiseStatisticConstView::getNumCols() const {
        return numCols_;
    }

    SparseLabelWiseStatisticView::SparseLabelWiseStatisticView(LilMatrix<Tuple<float64>>* statistics, uint32 numCols)
        : SparseLabelWiseStatisticConstView(statistics, numCols) {

    }

    void SparseLabelWiseStatisticView::clear() {
        statistics_->clear();
    }

    void SparseLabelWiseStatisticView::addToRow(uint32 row, const_iterator begin, const_iterator end, float64 weight) {
        // TODO Implement
    }

}
