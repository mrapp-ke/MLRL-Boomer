#include "mlrl/common/input/label_matrix_csr.hpp"

#include "mlrl/common/data/view_matrix_csr_binary.hpp"
#include "mlrl/common/prediction/probability_calibration_joint.hpp"
#include "mlrl/common/sampling/instance_sampling.hpp"
#include "mlrl/common/sampling/partition_sampling.hpp"
#include "mlrl/common/statistics/statistics_provider.hpp"
#include "mlrl/common/util/math.hpp"

/**
 * Implements row-wise read-only access to the labels of individual training examples that are stored in a pre-allocated
 * sparse matrix in the compressed sparse row (CSR) format.
 */
class CsrLabelMatrix final : public IterableBinarySparseMatrixDecorator<MatrixDecorator<BinaryCsrView>>,
                             public ICsrLabelMatrix {
    public:

        /**
         * @param indices   A pointer to an array of type `uint32`, shape `(numDenseElements)`, that stores the column
         *                  indices of all dense elements explicitly stored in the matrix
         * @param indptr    A pointer to an array of type `uint32`, shape `(numRows + 1)`, that stores the indices of
         *                  the first element in `indices` that corresponds to a certain row. The index at the last
         *                  position is equal to `numDenseElements`
         * @param numRows   The number of rows in the label matrix
         * @param numCols   The number of columns in the label matrix
         */
        CsrLabelMatrix(uint32* indices, uint32* indptr, uint32 numRows, uint32 numCols)
            : IterableBinarySparseMatrixDecorator<MatrixDecorator<BinaryCsrView>>(
                BinaryCsrView(indices, indptr, numRows, numCols)) {}

        bool isSparse() const override {
            return true;
        }

        uint32 getNumExamples() const override {
            return this->getNumRows();
        }

        uint32 getNumOutputs() const override {
            return this->getNumCols();
        }

        float32 calculateLabelCardinality() const override {
            uint32 numRows = this->getNumRows();
            float32 labelCardinality = 0;

            for (uint32 i = 0; i < numRows; i++) {
                index_const_iterator indicesBegin = this->indices_cbegin(i);
                index_const_iterator indicesEnd = this->indices_cend(i);
                uint32 numRelevantLabels = indicesEnd - indicesBegin;
                labelCardinality = iterativeArithmeticMean(i + 1, (float32) numRelevantLabels, labelCardinality);
            }

            return labelCardinality;
        }

        std::unique_ptr<LabelVector> createLabelVector(uint32 row) const override {
            index_const_iterator indexIterator = this->indices_cbegin(row);
            index_const_iterator indicesEnd = this->indices_cend(row);
            uint32 numElements = indicesEnd - indexIterator;
            std::unique_ptr<LabelVector> labelVectorPtr = std::make_unique<LabelVector>(numElements);
            LabelVector::iterator iterator = labelVectorPtr->begin();
            copyView(indexIterator, iterator, numElements);
            return labelVectorPtr;
        }

        std::unique_ptr<IStatisticsProvider> createStatisticsProvider(
          const IStatisticsProviderFactory& factory) const override {
            return factory.create(this->getView());
        }

        std::unique_ptr<IPartitionSampling> createPartitionSampling(
          const IClassificationPartitionSamplingFactory& factory) const override {
            return factory.create(this->getView());
        }

        std::unique_ptr<IInstanceSampling> createInstanceSampling(const IClassificationInstanceSamplingFactory& factory,
                                                                  const SinglePartition& partition,
                                                                  IStatistics& statistics) const override {
            return factory.create(this->getView(), partition, statistics);
        }

        std::unique_ptr<IInstanceSampling> createInstanceSampling(const IClassificationInstanceSamplingFactory& factory,
                                                                  BiPartition& partition,
                                                                  IStatistics& statistics) const override {
            return factory.create(this->getView(), partition, statistics);
        }

        std::unique_ptr<IMarginalProbabilityCalibrationModel> fitMarginalProbabilityCalibrationModel(
          const IMarginalProbabilityCalibrator& probabilityCalibrator, const SinglePartition& partition,
          const IStatistics& statistics) const override {
            return probabilityCalibrator.fitProbabilityCalibrationModel(partition, this->getView(), statistics);
        }

        std::unique_ptr<IMarginalProbabilityCalibrationModel> fitMarginalProbabilityCalibrationModel(
          const IMarginalProbabilityCalibrator& probabilityCalibrator, BiPartition& partition,
          const IStatistics& statistics) const override {
            return probabilityCalibrator.fitProbabilityCalibrationModel(partition, this->getView(), statistics);
        }

        std::unique_ptr<IJointProbabilityCalibrationModel> fitJointProbabilityCalibrationModel(
          const IJointProbabilityCalibrator& probabilityCalibrator, const SinglePartition& partition,
          const IStatistics& statistics) const override {
            return probabilityCalibrator.fitProbabilityCalibrationModel(partition, this->getView(), statistics);
        }

        std::unique_ptr<IJointProbabilityCalibrationModel> fitJointProbabilityCalibrationModel(
          const IJointProbabilityCalibrator& probabilityCalibrator, BiPartition& partition,
          const IStatistics& statistics) const override {
            return probabilityCalibrator.fitProbabilityCalibrationModel(partition, this->getView(), statistics);
        }
};

std::unique_ptr<ICsrLabelMatrix> createCsrLabelMatrix(uint32* indices, uint32* indptr, uint32 numRows, uint32 numCols) {
    return std::make_unique<CsrLabelMatrix>(indices, indptr, numRows, numCols);
}
