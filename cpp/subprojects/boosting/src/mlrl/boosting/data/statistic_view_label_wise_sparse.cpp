#include "mlrl/boosting/data/statistic_view_label_wise_sparse.hpp"

namespace boosting {

    SparseLabelWiseStatisticView::SparseLabelWiseStatisticView(ListOfLists<IndexedValue<Tuple<float64>>>&& valueView,
                                                               CContiguousView<uint32>&& indexView, uint32 numRows,
                                                               uint32 numCols)
        : SparseSetView<Tuple<float64>>(std::move(valueView), std::move(indexView), numRows, numCols) {}

    void SparseLabelWiseStatisticView::clear() {
        uint32 numRows = Matrix::numRows;

        for (uint32 i = 0; i < numRows; i++) {
            (*this)[i].clear();
        }
    }

    uint32 SparseLabelWiseStatisticView::getNumRows() const {
        return Matrix::numRows;
    }

    uint32 SparseLabelWiseStatisticView::getNumCols() const {
        return Matrix::numCols;
    }
}
