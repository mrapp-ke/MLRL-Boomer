#include "mlrl/boosting/data/statistic_view_label_wise_sparse.hpp"

namespace boosting {

    SparseLabelWiseStatisticView::SparseLabelWiseStatisticView(uint32 numCols,
                                                               SparseSetMatrix<Tuple<float64>>* statistics)
        : numCols_(numCols), statistics_(statistics) {}

    SparseLabelWiseStatisticView::const_iterator SparseLabelWiseStatisticView::cbegin(uint32 row) const {
        return statistics_->cbegin(row);
    }

    SparseLabelWiseStatisticView::const_iterator SparseLabelWiseStatisticView::cend(uint32 row) const {
        return statistics_->cend(row);
    }

    SparseLabelWiseStatisticView::const_row SparseLabelWiseStatisticView::operator[](uint32 row) const {
        return ((const SparseSetMatrix<Tuple<float64>>&) *statistics_)[row];
    }

    SparseLabelWiseStatisticView::row SparseLabelWiseStatisticView::operator[](uint32 row) {
        return (*statistics_)[row];
    }

    void SparseLabelWiseStatisticView::clear() {
        statistics_->clear();
    }

    uint32 SparseLabelWiseStatisticView::getNumRows() const {
        return statistics_->getNumRows();
    }

    uint32 SparseLabelWiseStatisticView::getNumCols() const {
        return numCols_;
    }
}
