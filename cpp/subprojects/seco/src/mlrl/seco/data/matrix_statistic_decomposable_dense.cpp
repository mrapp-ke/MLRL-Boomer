#include "mlrl/seco/data/matrix_statistic_decomposable_dense.hpp"

#include "mlrl/common/data/view_matrix_csr_binary.hpp"
#include "mlrl/seco/data/matrix_coverage_dense.hpp"

namespace seco {

    template<typename LabelMatrix>
    DenseDecomposableStatisticMatrix<LabelMatrix>::DenseDecomposableStatisticMatrix(
      const LabelMatrix& labelMatrix, std::unique_ptr<BinarySparseArrayVector> majorityLabelVectorPtr,
      std::unique_ptr<DenseCoverageMatrix> coverageMatrixPtr)
        : labelMatrix(labelMatrix), majorityLabelVectorPtr(std::move(majorityLabelVectorPtr)),
          coverageMatrixPtr(std::move(coverageMatrixPtr)) {}

    template<typename LabelMatrix>
    uint32 DenseDecomposableStatisticMatrix<LabelMatrix>::getNumRows() const {
        return labelMatrix.numRows;
    }

    template<typename LabelMatrix>
    uint32 DenseDecomposableStatisticMatrix<LabelMatrix>::getNumCols() const {
        return labelMatrix.numCols;
    }

    template<typename LabelMatrix>
    DenseDecomposableStatisticMatrix<LabelMatrix>::View DenseDecomposableStatisticMatrix<LabelMatrix>::getView() {
        return DenseDecomposableStatisticMatrix<LabelMatrix>::View(labelMatrix, *majorityLabelVectorPtr,
                                                                   *coverageMatrixPtr);
    }

    template class DenseDecomposableStatisticMatrix<CContiguousView<const uint8>>;
    template class DenseDecomposableStatisticMatrix<BinaryCsrView>;
}
