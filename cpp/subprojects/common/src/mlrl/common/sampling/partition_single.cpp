#include "mlrl/common/sampling/partition_single.hpp"

#include "mlrl/common/input/regression_matrix_row_wise.hpp"
#include "mlrl/common/prediction/probability_calibration_joint.hpp"
#include "mlrl/common/rule_refinement/feature_subspace.hpp"
#include "mlrl/common/rule_refinement/prediction_evaluated.hpp"
#include "mlrl/common/sampling/instance_sampling.hpp"
#include "mlrl/common/stopping/stopping_criterion.hpp"

SinglePartition::SinglePartition(uint32 numElements) : numElements_(numElements) {}

SinglePartition::const_iterator SinglePartition::cbegin() const {
    return IndexIterator();
}

SinglePartition::const_iterator SinglePartition::cend() const {
    return IndexIterator(numElements_);
}

uint32 SinglePartition::getNumElements() const {
    return numElements_;
}

std::unique_ptr<IStoppingCriterion> SinglePartition::createStoppingCriterion(const IStoppingCriterionFactory& factory) {
    return factory.create(*this);
}

std::unique_ptr<IInstanceSampling> SinglePartition::createInstanceSampling(
  const IClassificationInstanceSamplingFactory& factory, const IRowWiseLabelMatrix& labelMatrix,
  IStatistics& statistics, const EqualWeightVector& exampleWeights) {
    return labelMatrix.createInstanceSampling(factory, *this, statistics, exampleWeights);
}

std::unique_ptr<IInstanceSampling> SinglePartition::createInstanceSampling(
  const IClassificationInstanceSamplingFactory& factory, const IRowWiseLabelMatrix& labelMatrix,
  IStatistics& statistics, const DenseWeightVector<float32>& exampleWeights) {
    return labelMatrix.createInstanceSampling(factory, *this, statistics, exampleWeights);
}

std::unique_ptr<IInstanceSampling> SinglePartition::createInstanceSampling(
  const IRegressionInstanceSamplingFactory& factory, const IRowWiseRegressionMatrix& regressionMatrix,
  IStatistics& statistics, const EqualWeightVector& exampleWeights) {
    return regressionMatrix.createInstanceSampling(factory, *this, statistics, exampleWeights);
}

std::unique_ptr<IInstanceSampling> SinglePartition::createInstanceSampling(
  const IRegressionInstanceSamplingFactory& factory, const IRowWiseRegressionMatrix& regressionMatrix,
  IStatistics& statistics, const DenseWeightVector<float32>& exampleWeights) {
    return regressionMatrix.createInstanceSampling(factory, *this, statistics, exampleWeights);
}

Quality SinglePartition::evaluateOutOfSample(const IFeatureSubspace& featureSubspace, const CoverageMask& coverageMask,
                                             const IPrediction& head) {
    return featureSubspace.evaluateOutOfSample(*this, coverageMask, head);
}

void SinglePartition::recalculatePrediction(const IFeatureSubspace& featureSubspace, const CoverageMask& coverageMask,
                                            std::unique_ptr<IEvaluatedPrediction>& headPtr) {
    featureSubspace.recalculatePrediction(*this, coverageMask, headPtr);
}

std::unique_ptr<IMarginalProbabilityCalibrationModel> SinglePartition::fitMarginalProbabilityCalibrationModel(
  const IMarginalProbabilityCalibrator& probabilityCalibrator, const IRowWiseLabelMatrix& labelMatrix,
  const IStatistics& statistics) {
    return labelMatrix.fitMarginalProbabilityCalibrationModel(probabilityCalibrator, *this, statistics);
}

std::unique_ptr<IJointProbabilityCalibrationModel> SinglePartition::fitJointProbabilityCalibrationModel(
  const IJointProbabilityCalibrator& probabilityCalibrator, const IRowWiseLabelMatrix& labelMatrix,
  const IStatistics& statistics) {
    return labelMatrix.fitJointProbabilityCalibrationModel(probabilityCalibrator, *this, statistics);
}
