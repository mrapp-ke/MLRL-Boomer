#include "boosting/data/statistic_view_label_wise_sparse.hpp"


namespace boosting {

    SparseLabelWiseStatisticConstView::SparseLabelWiseStatisticConstView(LilMatrix<Tuple<float64>>* statistics,
                                                                         uint32 numCols)
        : statistics_(statistics), numCols_(numCols) {

    }

    SparseLabelWiseStatisticView::SparseLabelWiseStatisticView(LilMatrix<Tuple<float64>>* statistics, uint32 numCols)
        : SparseLabelWiseStatisticConstView(statistics, numCols) {

    }

}
