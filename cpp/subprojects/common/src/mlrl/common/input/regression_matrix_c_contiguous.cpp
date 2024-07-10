#include "mlrl/common/input/regression_matrix_c_contiguous.hpp"

#include "mlrl/common/data/matrix_dense.hpp"
#include "mlrl/common/data/view_matrix_c_contiguous.hpp"
#include "mlrl/common/sampling/instance_sampling.hpp"
#include "mlrl/common/sampling/partition_sampling.hpp"
#include "mlrl/common/statistics/statistics_provider.hpp"

/**
 * Implements random read-only access to the regression scores of individual training examples that are stored in a
 * pre-allocated C-contiguous array.
 */
class CContiguousRegressionMatrix final : public DenseMatrixDecorator<CContiguousView<const float32>>,
                                          public ICContiguousRegressionMatrix {
    public:

        /**
         * @param array     A pointer to a C-contiguous array of type `float32` that stores the regression scores
         * @param numRows   The number of rows in the regression matrix
         * @param numCols   The number of columns in the regression matrix
         */
        CContiguousRegressionMatrix(const float32* array, uint32 numRows, uint32 numCols)
            : DenseMatrixDecorator<CContiguousView<const float32>>(
                CContiguousView<const float32>(array, numRows, numCols)) {}

        bool isSparse() const override {
            return false;
        }

        uint32 getNumExamples() const override {
            return this->getNumRows();
        }

        uint32 getNumOutputs() const override {
            return this->getNumCols();
        }

        std::unique_ptr<IStatisticsProvider> createStatisticsProvider(
          const IStatisticsProviderFactory& factory) const override {
            // TODO return factory.create(this->getView());
            return nullptr;
        }

        std::unique_ptr<IPartitionSampling> createPartitionSampling(
          const IPartitionSamplingFactory& factory) const override {
            // TODO return factory.create(this->getView());
            return nullptr;
        }

        std::unique_ptr<IInstanceSampling> createInstanceSampling(const IInstanceSamplingFactory& factory,
                                                                  const SinglePartition& partition,
                                                                  IStatistics& statistics) const override {
            // TODO return factory.create(this->getView(), partition, statistics);
            return nullptr;
        }

        std::unique_ptr<IInstanceSampling> createInstanceSampling(const IInstanceSamplingFactory& factory,
                                                                  BiPartition& partition,
                                                                  IStatistics& statistics) const override {
            // TODO return factory.create(this->getView(), partition, statistics);
            return nullptr;
        }
};

std::unique_ptr<ICContiguousRegressionMatrix> createCContiguousRegressionMatrix(const float32* array, uint32 numRows,
                                                                                uint32 numCols) {
    return std::make_unique<CContiguousRegressionMatrix>(array, numRows, numCols);
}
