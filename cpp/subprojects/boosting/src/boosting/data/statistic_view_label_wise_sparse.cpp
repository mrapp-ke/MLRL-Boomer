#include "boosting/data/statistic_view_label_wise_sparse.hpp"


namespace boosting {

    SparseLabelWiseStatisticConstView::SparseLabelWiseStatisticConstView(uint32 numCols,
                                                                         SparseSetMatrix<Tuple<float64>>* statistics)
        : numCols_(numCols), statistics_(statistics) {

    }

    SparseLabelWiseStatisticConstView::const_iterator SparseLabelWiseStatisticConstView::row_cbegin(uint32 row) const {
        return statistics_->row_cbegin(row);
    }

    SparseLabelWiseStatisticConstView::const_iterator SparseLabelWiseStatisticConstView::row_cend(uint32 row) const {
        return statistics_->row_cend(row);
    }

    SparseLabelWiseStatisticConstView::const_row SparseLabelWiseStatisticConstView::getRow(uint32 row) const {
        const SparseSetMatrix<Tuple<float64>>& ref = *statistics_;
        return ref.getRow(row);
    }

    uint32 SparseLabelWiseStatisticConstView::getNumRows() const {
        return statistics_->getNumRows();
    }

    uint32 SparseLabelWiseStatisticConstView::getNumCols() const {
        return numCols_;
    }

    SparseLabelWiseStatisticView::SparseLabelWiseStatisticView(uint32 numCols,
                                                               SparseSetMatrix<Tuple<float64>>* statistics)
        : SparseLabelWiseStatisticConstView(numCols, statistics) {

    }

    SparseLabelWiseStatisticView::row SparseLabelWiseStatisticView::getRow(uint32 row) {
        return statistics_->getRow(row);
    }

    void SparseLabelWiseStatisticView::clear() {
        statistics_->clear();
    }

}
