#include "mlrl/boosting/data/view_histogram_label_wise_sparse.hpp"

namespace boosting {

    SparseLabelWiseHistogramView::SparseLabelWiseHistogramView(uint32 numRows, uint32 numCols)
        : CompositeMatrix<AllocatedCContiguousView<Triple<float64>>, AllocatedVector<float64>>(
          AllocatedCContiguousView<Triple<float64>>(numRows, numCols), AllocatedVector<float64>(numRows, true), numRows,
          numCols) {}

    SparseLabelWiseHistogramView::SparseLabelWiseHistogramView(SparseLabelWiseHistogramView&& other)
        : CompositeMatrix<AllocatedCContiguousView<Triple<float64>>, AllocatedVector<float64>>(std::move(other)) {}

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
