#include "mlrl/boosting/data/histogram_view_label_wise_sparse.hpp"

#include "statistic_vector_label_wise_sparse_common.hpp"

namespace boosting {

    SparseLabelWiseHistogramView::SparseLabelWiseHistogramView(uint32 numRows, uint32 numCols,
                                                               Triple<float64>* statistics, float64* weights)
        : numRows_(numRows), numCols_(numCols), statistics_(statistics), weights_(weights) {}

    SparseLabelWiseHistogramView::const_iterator SparseLabelWiseHistogramView::cbegin(uint32 row) const {
        return &statistics_[row * numCols_];
    }

    SparseLabelWiseHistogramView::const_iterator SparseLabelWiseHistogramView::cend(uint32 row) const {
        return &statistics_[(row + 1) * numCols_];
    }

    SparseLabelWiseHistogramView::weight_const_iterator SparseLabelWiseHistogramView::weights_cbegin() const {
        return weights_;
    }

    SparseLabelWiseHistogramView::weight_const_iterator SparseLabelWiseHistogramView::weights_cend() const {
        return &weights_[numRows_];
    }

    void SparseLabelWiseHistogramView::clear() {
        setViewToZeros(weights_, numRows_);
        setViewToZeros(statistics_, numRows_ * numCols_);
    }

    void SparseLabelWiseHistogramView::addToRow(uint32 row, SparseSetView<Tuple<float64>>::const_iterator begin,
                                                SparseSetView<Tuple<float64>>::const_iterator end, float64 weight) {
        if (weight != 0) {
            weights_[row] += weight;
            addToSparseLabelWiseStatisticVector(&statistics_[row * numCols_], begin, end, weight);
        }
    }

    uint32 SparseLabelWiseHistogramView::getNumRows() const {
        return numRows_;
    }

    uint32 SparseLabelWiseHistogramView::getNumCols() const {
        return numCols_;
    }
}
