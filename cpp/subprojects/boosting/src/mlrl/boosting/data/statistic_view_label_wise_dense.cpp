#include "mlrl/boosting/data/statistic_view_label_wise_dense.hpp"

#include "mlrl/common/util/view_functions.hpp"

namespace boosting {

    DenseLabelWiseStatisticView::DenseLabelWiseStatisticView(uint32 numRows, uint32 numCols, Tuple<float64>* statistics)
        : CContiguousView<Tuple<float64>>(statistics, numRows, numCols) {}

    void DenseLabelWiseStatisticView::clear() {
        setViewToZeros(DenseMatrix::array, Matrix::numRows * Matrix::numCols);
    }

    void DenseLabelWiseStatisticView::addToRow(uint32 row, value_const_iterator begin, value_const_iterator end,
                                               float64 weight) {
        addToView(CContiguousView::values_begin(row), begin, Matrix::numCols, weight);
    }

    uint32 DenseLabelWiseStatisticView::getNumRows() const {
        return Matrix::numRows;
    }

    uint32 DenseLabelWiseStatisticView::getNumCols() const {
        return Matrix::numCols;
    }
}
