#include "mlrl/common/input/label_matrix_c_contiguous.hpp"

#include "mlrl/common/data/matrix_dense.hpp"
#include "mlrl/common/data/view_matrix_c_contiguous.hpp"
#include "mlrl/common/prediction/probability_calibration_joint.hpp"
#include "mlrl/common/sampling/instance_sampling.hpp"
#include "mlrl/common/sampling/partition_sampling.hpp"
#include "mlrl/common/statistics/statistics_provider.hpp"
#include "mlrl/common/util/math.hpp"

/**
 * Implements random read-only access to the labels of individual training examples that are stored in a pre-allocated
 * C-contiguous array.
 */
class CContiguousLabelMatrix final : public DenseMatrixDecorator<CContiguousView<const uint8>>,
                                     public ICContiguousLabelMatrix {
    public:

        /**
         * @param array     A pointer to a C-contiguous array of type `uint8` that stores the labels
         * @param numRows   The number of rows in the label matrix
         * @param numCols   The number of columns in the label matrix
         */
        CContiguousLabelMatrix(const uint8* array, uint32 numRows, uint32 numCols)
            : DenseMatrixDecorator<CContiguousView<const uint8>>(
                CContiguousView<const uint8>(array, numRows, numCols)) {}

        bool isSparse() const override {
            return false;
        }

        uint32 getNumExamples() const override {
            return this->getNumRows();
        }

        uint32 getNumOutputs() const override {
            return this->getNumCols();
        }

        float32 calculateLabelCardinality() const override {
            uint32 numRows = this->getNumRows();
            uint32 numCols = this->getNumCols();
            float32 labelCardinality = 0;

            for (uint32 i = 0; i < numRows; i++) {
                value_const_iterator labelIterator = this->values_cbegin(i);
                uint32 numRelevantLabels = 0;

                for (uint32 j = 0; j < numCols; j++) {
                    if (labelIterator[j]) {
                        numRelevantLabels++;
                    }
                }

                labelCardinality = util::iterativeArithmeticMean(i + 1, (float32) numRelevantLabels, labelCardinality);
            }

            return labelCardinality;
        }

        std::unique_ptr<LabelVector> createLabelVector(uint32 row) const override {
            uint32 numCols = this->getNumCols();
            std::unique_ptr<ResizableBinarySparseArrayVector> labelVectorPtr =
              std::make_unique<ResizableBinarySparseArrayVector>(numCols);
            ResizableBinarySparseArrayVector::iterator iterator = labelVectorPtr->begin();
            value_const_iterator labelIterator = this->values_cbegin(row);
            uint32 n = 0;

            for (uint32 i = 0; i < numCols; i++) {
                if (labelIterator[i]) {
                    iterator[n] = i;
                    n++;
                }
            }

            labelVectorPtr->setNumElements(n, true);
            return std::make_unique<LabelVector>(std::move(labelVectorPtr->getView()));
        }

        std::unique_ptr<IStatisticsProvider> createStatisticsProvider(
          const IClassificationStatisticsProviderFactory& factory) const override {
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

std::unique_ptr<ICContiguousLabelMatrix> createCContiguousLabelMatrix(const uint8* array, uint32 numRows,
                                                                      uint32 numCols) {
    return std::make_unique<CContiguousLabelMatrix>(array, numRows, numCols);
}
