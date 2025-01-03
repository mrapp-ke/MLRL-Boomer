#include "mlrl/boosting/data/view_statistic_decomposable_bit.hpp"

#include "mlrl/boosting/util/math.hpp"

namespace boosting {

    BitDecomposableStatisticView::BitDecomposableStatisticView(uint32 numRows, uint32 numCols, uint32 numBits)
        : CompositeMatrix<AllocatedBitMatrix<uint32>, AllocatedBitMatrix<uint32>>(
            AllocatedBitMatrix<uint32>(numRows, numCols, numBits),
            AllocatedBitMatrix<uint32>(numRows, util::triangularNumber(numCols), numBits), numRows, numCols) {}

    BitDecomposableStatisticView::BitDecomposableStatisticView(BitDecomposableStatisticView&& other)
        : CompositeMatrix<AllocatedBitMatrix<uint32>, AllocatedBitMatrix<uint32>>(std::move(other)) {}

}
