#include "boosting/data/statistic_view_label_wise_sparse.hpp"


namespace boosting {

    SparseLabelWiseStatisticConstView::SparseLabelWiseStatisticConstView(LilMatrix<Tuple<float64>>* statistics,
                                                                         uint32 numCols)
        : statistics_(statistics), numCols_(numCols) {

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

}
