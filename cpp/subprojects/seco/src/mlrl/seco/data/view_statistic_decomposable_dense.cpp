#include "mlrl/seco/data/view_statistic_decomposable_dense.hpp"

#include "mlrl/common/data/view_matrix_csr_binary.hpp"
#include "mlrl/seco/data/matrix_coverage_dense.hpp"

namespace seco {

    template<typename LabelMatrix, typename CoverageMatrix>
    DenseConfusionMatrixView<LabelMatrix, CoverageMatrix>::DenseConfusionMatrixView(
      const LabelMatrix& labelMatrix, std::unique_ptr<BinarySparseArrayVector> majorityLabelVectorPtr,
      std::unique_ptr<CoverageMatrix> coverageMatrixPtr)
        : labelMatrix(labelMatrix), majorityLabelVectorPtr(std::move(majorityLabelVectorPtr)),
          coverageMatrixPtr(std::move(coverageMatrixPtr)) {}

    template class DenseDecomposableStatisticView<CContiguousView<const uint8>, DenseCoverageMatrix>;
    template class DenseDecomposableStatisticView<BinaryCsrView, DenseCoverageMatrix>;
}
