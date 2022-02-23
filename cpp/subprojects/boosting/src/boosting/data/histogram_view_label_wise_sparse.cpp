#include "boosting/data/histogram_view_label_wise_sparse.hpp"
#include "statistic_vector_label_wise_sparse_common.hpp"


namespace boosting {

    SparseLabelWiseHistogramConstView::SparseLabelWiseHistogramConstView(uint32 numCols,
                                                                         LilMatrix<Triple<float64>>* histogram)
        : numCols_(numCols), histogram_(histogram) {

    }

    SparseLabelWiseHistogramConstView::const_iterator SparseLabelWiseHistogramConstView::row_cbegin(uint32 row) const {
        return histogram_->getRow(row).cbegin();
    }

    SparseLabelWiseHistogramConstView::const_iterator SparseLabelWiseHistogramConstView::row_cend(uint32 row) const {
        return histogram_->getRow(row).cend();
    }

    const SparseLabelWiseHistogramConstView::Row& SparseLabelWiseHistogramConstView::getRow(uint32 row) const {
        return histogram_->getRow(row);
    }

    uint32 SparseLabelWiseHistogramConstView::getNumRows() const {
        return histogram_->getNumRows();
    }

    uint32 SparseLabelWiseHistogramConstView::getNumCols() const {
        return numCols_;
    }

    SparseLabelWiseHistogramView::SparseLabelWiseHistogramView(uint32 numCols, LilMatrix<Triple<float64>>* histogram)
        : SparseLabelWiseHistogramConstView(numCols, histogram) {

    }

    SparseLabelWiseHistogramView::Row& SparseLabelWiseHistogramView::getRow(uint32 row) {
        return histogram_->getRow(row);
    }

    void SparseLabelWiseHistogramView::clear() {
        histogram_->clear();
    }

    void SparseLabelWiseHistogramView::addToRow(uint32 row, SparseLabelWiseStatisticConstView::const_iterator begin,
                                                SparseLabelWiseStatisticConstView::const_iterator end, float64 weight) {
        addToSparseLabelWiseStatisticVector<Tuple<float64>, SparseLabelWiseStatisticConstView::const_iterator>(
            histogram_->getRow(row), begin, end, weight);
    }

}
