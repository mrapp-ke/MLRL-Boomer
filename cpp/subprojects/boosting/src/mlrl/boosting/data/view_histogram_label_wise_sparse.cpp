#include "mlrl/boosting/data/view_histogram_label_wise_sparse.hpp"

namespace boosting {

    SparseLabelWiseHistogramView::SparseLabelWiseHistogramView(CContiguousView<Triple<float64>>&& firstView,
                                                               Vector<float64>&& secondView, uint32 numRows,
                                                               uint32 numCols)
        : CompositeMatrix<CContiguousView<Triple<float64>>, Vector<float64>>(std::move(firstView),
                                                                             std::move(secondView), numRows, numCols) {}

    SparseLabelWiseHistogramView::value_const_iterator SparseLabelWiseHistogramView::values_cbegin(uint32 row) const {
        return CompositeView::firstView.values_cbegin(row);
    }

    SparseLabelWiseHistogramView::value_const_iterator SparseLabelWiseHistogramView::values_cend(uint32 row) const {
        return CompositeView::firstView.values_cend(row);
    }

    SparseLabelWiseHistogramView::weight_const_iterator SparseLabelWiseHistogramView::weights_cbegin() const {
        return CompositeView::secondView.cbegin();
    }

    SparseLabelWiseHistogramView::weight_const_iterator SparseLabelWiseHistogramView::weights_cend() const {
        return CompositeView::secondView.cend();
    }
}
