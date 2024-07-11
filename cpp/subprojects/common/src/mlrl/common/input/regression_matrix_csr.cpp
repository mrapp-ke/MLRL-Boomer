#include "mlrl/common/input/regression_matrix_csr.hpp"

#include "mlrl/common/data/view_matrix_csr.hpp"
#include "mlrl/common/sampling/instance_sampling.hpp"
#include "mlrl/common/sampling/partition_sampling.hpp"
#include "mlrl/common/statistics/statistics_provider.hpp"

/**
 * Implements row-wise read-only access to the regression scores of individual training examples that are stored in a
 * pre-allocated sparse matrix in the compressed sparse row (CSR) format.
 */
class CsrRegressionMatrix final : public IterableBinarySparseMatrixDecorator<MatrixDecorator<CsrView<const float32>>>,
                                  public ICsrRegressionMatrix {
    public:

        /**
         * @param values    A pointer to an array of type `float32`, shape `(numDenseElements)`, that stores the values
         *                  of all dense elements explicitly stored in the matrix
         * @param indices   A pointer to an array of type `uint32`, shape `(numDenseElements)`, that stores the column
         *                  indices of all dense elements explicitly stored in the matrix
         * @param indptr    A pointer to an array of type `uint32`, shape `(numRows + 1)`, that stores the indices of
         *                  the first element in `indices` that corresponds to a certain row. The index at the last
         *                  position is equal to `numDenseElements`
         * @param numRows   The number of rows in the regression matrix
         * @param numCols   The number of columns in the regression matrix
         */
        CsrRegressionMatrix(float32* values, uint32* indices, uint32* indptr, uint32 numRows, uint32 numCols)
            : IterableBinarySparseMatrixDecorator<MatrixDecorator<CsrView<const float32>>>(
                CsrView<const float32>(values, indices, indptr, numRows, numCols)) {}

        bool isSparse() const override {
            return true;
        }

        uint32 getNumExamples() const override {
            return this->getNumRows();
        }

        uint32 getNumOutputs() const override {
            return this->getNumCols();
        }

        std::unique_ptr<IStatisticsProvider> createStatisticsProvider(
          const IStatisticsProviderFactory& factory) const override {
            return factory.create(this->getView());
        }

        std::unique_ptr<IPartitionSampling> createPartitionSampling(
          const IRegressionPartitionSamplingFactory& factory) const override {
            return factory.create(this->getView());
        }

        std::unique_ptr<IInstanceSampling> createInstanceSampling(const IRegressionInstanceSamplingFactory& factory,
                                                                  const SinglePartition& partition,
                                                                  IStatistics& statistics) const override {
            return factory.create(this->getView(), partition, statistics);
        }

        std::unique_ptr<IInstanceSampling> createInstanceSampling(const IRegressionInstanceSamplingFactory& factory,
                                                                  BiPartition& partition,
                                                                  IStatistics& statistics) const override {
            return factory.create(this->getView(), partition, statistics);
        }
};

std::unique_ptr<ICsrRegressionMatrix> createCsrRegressionMatrix(float32* values, uint32* indices, uint32* indptr,
                                                                uint32 numRows, uint32 numCols) {
    return std::make_unique<CsrRegressionMatrix>(values, indices, indptr, numRows, numCols);
}
