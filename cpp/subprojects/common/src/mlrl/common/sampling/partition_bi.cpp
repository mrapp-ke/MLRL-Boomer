#include "mlrl/common/sampling/partition_bi.hpp"

#include "mlrl/common/input/regression_matrix_row_wise.hpp"
#include "mlrl/common/prediction/probability_calibration_joint.hpp"
#include "mlrl/common/rule_refinement/feature_subspace.hpp"
#include "mlrl/common/rule_refinement/prediction_evaluated.hpp"
#include "mlrl/common/sampling/instance_sampling.hpp"
#include "mlrl/common/stopping/stopping_criterion.hpp"

#include <algorithm>

BiPartition::BiPartition(uint32 numFirst, uint32 numSecond)
    : VectorDecorator<AllocatedVector<uint32>>(AllocatedVector<uint32>(numFirst + numSecond)), numFirst_(numFirst),
      firstSorted_(false), secondSorted_(false) {}

BiPartition::iterator BiPartition::first_begin() {
    return this->view.begin();
}

BiPartition::iterator BiPartition::first_end() {
    return &this->view.array[numFirst_];
}

BiPartition::const_iterator BiPartition::first_cbegin() const {
    return this->view.cbegin();
}

BiPartition::const_iterator BiPartition::first_cend() const {
    return &this->view.array[numFirst_];
}

BiPartition::iterator BiPartition::second_begin() {
    return &this->view.array[numFirst_];
}

BiPartition::iterator BiPartition::second_end() {
    return &this->view.array[this->view.numElements];
}

BiPartition::const_iterator BiPartition::second_cbegin() const {
    return &this->view.array[numFirst_];
}

BiPartition::const_iterator BiPartition::second_cend() const {
    return &this->view.array[this->view.numElements];
}

uint32 BiPartition::getNumFirst() const {
    return numFirst_;
}

uint32 BiPartition::getNumSecond() const {
    return this->view.numElements - numFirst_;
}

void BiPartition::sortFirst() {
    if (!firstSorted_) {
        std::sort(this->first_begin(), this->first_end(), std::less<uint32>());
        firstSorted_ = true;
    }
}

void BiPartition::sortSecond() {
    if (!secondSorted_) {
        std::sort(this->second_begin(), this->second_end(), std::less<uint32>());
        secondSorted_ = true;
    }
}

std::unique_ptr<IStoppingCriterion> BiPartition::createStoppingCriterion(const IStoppingCriterionFactory& factory) {
    return factory.create(*this);
}

std::unique_ptr<IInstanceSampling> BiPartition::createInstanceSampling(
  const IClassificationInstanceSamplingFactory& factory, const IRowWiseLabelMatrix& labelMatrix,
  IStatistics& statistics, const EqualWeightVector& exampleWeights) {
    return labelMatrix.createInstanceSampling(factory, *this, statistics, exampleWeights);
}

std::unique_ptr<IInstanceSampling> BiPartition::createInstanceSampling(
  const IRegressionInstanceSamplingFactory& factory, const IRowWiseRegressionMatrix& regressionMatrix,
  IStatistics& statistics, const EqualWeightVector& exampleWeights) {
    return regressionMatrix.createInstanceSampling(factory, *this, statistics, exampleWeights);
}

std::unique_ptr<IInstanceSampling> BiPartition::createInstanceSampling(
  const IClassificationInstanceSamplingFactory& factory, const IRowWiseLabelMatrix& labelMatrix,
  IStatistics& statistics, const DenseWeightVector<float32>& exampleWeights) {
    return labelMatrix.createInstanceSampling(factory, *this, statistics, exampleWeights);
}

std::unique_ptr<IInstanceSampling> BiPartition::createInstanceSampling(
  const IRegressionInstanceSamplingFactory& factory, const IRowWiseRegressionMatrix& regressionMatrix,
  IStatistics& statistics, const DenseWeightVector<float32>& exampleWeights) {
    return regressionMatrix.createInstanceSampling(factory, *this, statistics, exampleWeights);
}

Quality BiPartition::evaluateOutOfSample(const IFeatureSubspace& featureSubspace, const CoverageMask& coverageMask,
                                         const IPrediction& head) {
    return featureSubspace.evaluateOutOfSample(*this, coverageMask, head);
}

void BiPartition::recalculatePrediction(const IFeatureSubspace& featureSubspace, const CoverageMask& coverageMask,
                                        std::unique_ptr<IEvaluatedPrediction>& headPtr) {
    featureSubspace.recalculatePrediction(*this, coverageMask, headPtr);
}

std::unique_ptr<IMarginalProbabilityCalibrationModel> BiPartition::fitMarginalProbabilityCalibrationModel(
  const IMarginalProbabilityCalibrator& probabilityCalibrator, const IRowWiseLabelMatrix& labelMatrix,
  const IStatistics& statistics) {
    return labelMatrix.fitMarginalProbabilityCalibrationModel(probabilityCalibrator, *this, statistics);
}

std::unique_ptr<IJointProbabilityCalibrationModel> BiPartition::fitJointProbabilityCalibrationModel(
  const IJointProbabilityCalibrator& probabilityCalibrator, const IRowWiseLabelMatrix& labelMatrix,
  const IStatistics& statistics) {
    return labelMatrix.fitJointProbabilityCalibrationModel(probabilityCalibrator, *this, statistics);
}
