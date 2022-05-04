#include "boosting/data/histogram_view_label_wise_sparse.hpp"
#include "common/data/arrays.hpp"
#include <iostream>  // TODO Remove


namespace boosting {

    SparseLabelWiseHistogramConstView::SparseLabelWiseHistogramConstView(uint32 numCols,
                                                                         LilMatrix<Triple<float64>>* histogram,
                                                                         DenseVector<float64>* weights)
        : numCols_(numCols), histogram_(histogram), weights_(weights) {

    }

    SparseLabelWiseHistogramConstView::const_iterator SparseLabelWiseHistogramConstView::row_cbegin(uint32 row) const {
        return histogram_->row_cbegin(row);
    }

    SparseLabelWiseHistogramConstView::const_iterator SparseLabelWiseHistogramConstView::row_cend(uint32 row) const {
        return histogram_->row_cend(row);
    }

    const SparseLabelWiseHistogramConstView::Row SparseLabelWiseHistogramConstView::getRow(uint32 row) const {
        return histogram_->getRow(row);
    }

    const float64 SparseLabelWiseHistogramConstView::getWeight(uint32 row) const {
        return (*weights_)[row];
    }

    uint32 SparseLabelWiseHistogramConstView::getNumRows() const {
        return histogram_->getNumRows();
    }

    uint32 SparseLabelWiseHistogramConstView::getNumCols() const {
        return numCols_;
    }

    SparseLabelWiseHistogramView::SparseLabelWiseHistogramView(uint32 numCols, LilMatrix<Triple<float64>>* histogram,
                                                               DenseVector<float64>* weights)
        : SparseLabelWiseHistogramConstView(numCols, histogram, weights) {

    }

    SparseLabelWiseHistogramView::Row SparseLabelWiseHistogramView::getRow(uint32 row) {
        return histogram_->getRow(row);
    }

    void SparseLabelWiseHistogramView::clear() {
        // TODO Implement
        std::cout << "SparseLabelWiseHistogramView::clear()\n";
        std::exit(-1);
    }

    void SparseLabelWiseHistogramView::addToRow(uint32 row, SparseLabelWiseStatisticConstView::const_iterator begin,
                                                SparseLabelWiseStatisticConstView::const_iterator end, float64 weight) {
        // TODO Implement
        std::cout << "SparseLabelWiseHistogramView::addToRow()\n";
        std::exit(-1);
    }

}
