#include "mlrl/common/input/label_matrix_c_contiguous.hpp"

#include "mlrl/common/prediction/probability_calibration_joint.hpp"
#include "mlrl/common/sampling/instance_sampling.hpp"
#include "mlrl/common/sampling/partition_sampling.hpp"
#include "mlrl/common/statistics/statistics_provider.hpp"
#include "mlrl/common/util/math.hpp"

CContiguousLabelMatrix::CContiguousLabelMatrix(const uint8* array, uint32 numRows, uint32 numCols)
    : IterableDenseMatrixDecorator<MatrixDecorator<CContiguousView<const uint8>>>(
      CContiguousView<const uint8>(array, numRows, numCols)) {}

bool CContiguousLabelMatrix::isSparse() const {
    return false;
}

uint32 CContiguousLabelMatrix::getNumExamples() const {
    return this->getNumRows();
}

uint32 CContiguousLabelMatrix::getNumLabels() const {
    return this->getNumCols();
}

float32 CContiguousLabelMatrix::calculateLabelCardinality() const {
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

        labelCardinality = iterativeArithmeticMean(i + 1, (float32) numRelevantLabels, labelCardinality);
    }

    return labelCardinality;
}

std::unique_ptr<LabelVector> CContiguousLabelMatrix::createLabelVector(uint32 row) const {
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

std::unique_ptr<IStatisticsProvider> CContiguousLabelMatrix::createStatisticsProvider(
  const IStatisticsProviderFactory& factory) const {
    return factory.create(this->getView());
}

std::unique_ptr<IPartitionSampling> CContiguousLabelMatrix::createPartitionSampling(
  const IPartitionSamplingFactory& factory) const {
    return factory.create(this->getView());
}

std::unique_ptr<IInstanceSampling> CContiguousLabelMatrix::createInstanceSampling(
  const IInstanceSamplingFactory& factory, const SinglePartition& partition, IStatistics& statistics) const {
    return factory.create(this->getView(), partition, statistics);
}

std::unique_ptr<IInstanceSampling> CContiguousLabelMatrix::createInstanceSampling(
  const IInstanceSamplingFactory& factory, BiPartition& partition, IStatistics& statistics) const {
    return factory.create(this->getView(), partition, statistics);
}

std::unique_ptr<IMarginalProbabilityCalibrationModel> CContiguousLabelMatrix::fitMarginalProbabilityCalibrationModel(
  const IMarginalProbabilityCalibrator& probabilityCalibrator, const SinglePartition& partition,
  const IStatistics& statistics) const {
    return probabilityCalibrator.fitProbabilityCalibrationModel(partition, this->getView(), statistics);
}

std::unique_ptr<IMarginalProbabilityCalibrationModel> CContiguousLabelMatrix::fitMarginalProbabilityCalibrationModel(
  const IMarginalProbabilityCalibrator& probabilityCalibrator, BiPartition& partition,
  const IStatistics& statistics) const {
    return probabilityCalibrator.fitProbabilityCalibrationModel(partition, this->getView(), statistics);
}

std::unique_ptr<IJointProbabilityCalibrationModel> CContiguousLabelMatrix::fitJointProbabilityCalibrationModel(
  const IJointProbabilityCalibrator& probabilityCalibrator, const SinglePartition& partition,
  const IStatistics& statistics) const {
    return probabilityCalibrator.fitProbabilityCalibrationModel(partition, this->getView(), statistics);
}

std::unique_ptr<IJointProbabilityCalibrationModel> CContiguousLabelMatrix::fitJointProbabilityCalibrationModel(
  const IJointProbabilityCalibrator& probabilityCalibrator, BiPartition& partition,
  const IStatistics& statistics) const {
    return probabilityCalibrator.fitProbabilityCalibrationModel(partition, this->getView(), statistics);
}

std::unique_ptr<ICContiguousLabelMatrix> createCContiguousLabelMatrix(const uint8* array, uint32 numRows,
                                                                      uint32 numCols) {
    return std::make_unique<CContiguousLabelMatrix>(array, numRows, numCols);
}
