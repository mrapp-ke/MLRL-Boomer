#include "mlrl/seco/data/matrix_statistic_decomposable_dense.hpp"

#include "mlrl/common/data/view_matrix_csr_binary.hpp"
#include "mlrl/seco/data/matrix_coverage_dense.hpp"

namespace seco {

    template<typename LabelMatrix, typename CoverageMatrix>
    DenseDecomposableStatisticMatrix<LabelMatrix, CoverageMatrix>::DenseDecomposableStatisticMatrix(
      const LabelMatrix& labelMatrix, std::unique_ptr<BinarySparseArrayVector> majorityLabelVectorPtr,
      std::unique_ptr<CoverageMatrix> coverageMatrixPtr)
        : labelMatrix(labelMatrix), majorityLabelVectorPtr(std::move(majorityLabelVectorPtr)),
          coverageMatrixPtr(std::move(coverageMatrixPtr)) {}

    template<typename LabelMatrix, typename CoverageMatrix>
    uint32 DenseDecomposableStatisticMatrix<LabelMatrix, CoverageMatrix>::getNumRows() const {
        return labelMatrix.numRows;
    }

    template<typename LabelMatrix, typename CoverageMatrix>
    uint32 DenseDecomposableStatisticMatrix<LabelMatrix, CoverageMatrix>::getNumCols() const {
        return labelMatrix.numCols;
    }

    template<typename LabelMatrix, typename CoverageMatrix>
    DenseDecomposableStatisticMatrix<LabelMatrix, CoverageMatrix>::View
      DenseDecomposableStatisticMatrix<LabelMatrix, CoverageMatrix>::getView() {
        return DenseDecomposableStatisticMatrix<LabelMatrix, CoverageMatrix>::View(labelMatrix, *majorityLabelVectorPtr,
                                                                                   *coverageMatrixPtr);
    }

    template class DenseDecomposableStatisticMatrix<CContiguousView<const uint8>, DenseCoverageMatrix>;
    template class DenseDecomposableStatisticMatrix<BinaryCsrView, DenseCoverageMatrix>;
}
