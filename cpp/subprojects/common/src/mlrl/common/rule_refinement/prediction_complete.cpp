#include "mlrl/common/rule_refinement/prediction_complete.hpp"

#include "mlrl/common/model/head_complete.hpp"
#include "mlrl/common/post_processing/post_processor.hpp"
#include "mlrl/common/rule_refinement/rule_refinement.hpp"
#include "mlrl/common/statistics/statistics.hpp"

CompletePrediction::CompletePrediction(uint32 numElements)
    : VectorDecorator<AllocatedVector<float64>>(AllocatedVector<float64>(numElements)), indexVector_(numElements) {}

CompletePrediction::value_iterator CompletePrediction::values_begin() {
    return this->view_.begin();
}

CompletePrediction::value_iterator CompletePrediction::values_end() {
    return this->view_.end();
}

CompletePrediction::value_const_iterator CompletePrediction::values_cbegin() const {
    return this->view_.cbegin();
}

CompletePrediction::value_const_iterator CompletePrediction::values_cend() const {
    return this->view_.cend();
}

CompletePrediction::index_const_iterator CompletePrediction::indices_cbegin() const {
    return indexVector_.cbegin();
}

CompletePrediction::index_const_iterator CompletePrediction::indices_cend() const {
    return indexVector_.cend();
}

uint32 CompletePrediction::getNumElements() const {
    return VectorDecorator<AllocatedVector<float64>>::getNumElements();
}

void CompletePrediction::sort() {}

void CompletePrediction::postProcess(const IPostProcessor& postProcessor) {
    postProcessor.postProcess(this->values_begin(), this->values_end());
}

void CompletePrediction::set(View<float64>::const_iterator begin, View<float64>::const_iterator end) {
    copyView(begin, this->view_.begin(), this->getNumElements());
}

void CompletePrediction::set(DenseBinnedVector<float64>::const_iterator begin,
                             DenseBinnedVector<float64>::const_iterator end) {
    copyView(begin, this->view_.begin(), this->getNumElements());
}

bool CompletePrediction::isPartial() const {
    return false;
}

uint32 CompletePrediction::getIndex(uint32 pos) const {
    return indexVector_.getIndex(pos);
}

std::unique_ptr<IStatisticsSubset> CompletePrediction::createStatisticsSubset(const IStatistics& statistics,
                                                                              const EqualWeightVector& weights) const {
    return statistics.createSubset(indexVector_, weights);
}

std::unique_ptr<IStatisticsSubset> CompletePrediction::createStatisticsSubset(const IStatistics& statistics,
                                                                              const BitWeightVector& weights) const {
    return statistics.createSubset(indexVector_, weights);
}

std::unique_ptr<IStatisticsSubset> CompletePrediction::createStatisticsSubset(
  const IStatistics& statistics, const DenseWeightVector<uint32>& weights) const {
    return statistics.createSubset(indexVector_, weights);
}

std::unique_ptr<IStatisticsSubset> CompletePrediction::createStatisticsSubset(
  const IStatistics& statistics, const OutOfSampleWeightVector<EqualWeightVector>& weights) const {
    return statistics.createSubset(indexVector_, weights);
}

std::unique_ptr<IStatisticsSubset> CompletePrediction::createStatisticsSubset(
  const IStatistics& statistics, const OutOfSampleWeightVector<BitWeightVector>& weights) const {
    return statistics.createSubset(indexVector_, weights);
}

std::unique_ptr<IStatisticsSubset> CompletePrediction::createStatisticsSubset(
  const IStatistics& statistics, const OutOfSampleWeightVector<DenseWeightVector<uint32>>& weights) const {
    return statistics.createSubset(indexVector_, weights);
}

std::unique_ptr<IRuleRefinement> CompletePrediction::createRuleRefinement(IThresholdsSubset& thresholdsSubset,
                                                                          uint32 featureIndex) const {
    return indexVector_.createRuleRefinement(thresholdsSubset, featureIndex);
}

void CompletePrediction::apply(IStatistics& statistics, uint32 statisticIndex) const {
    statistics.applyPrediction(statisticIndex, *this);
}

void CompletePrediction::revert(IStatistics& statistics, uint32 statisticIndex) const {
    statistics.revertPrediction(statisticIndex, *this);
}

std::unique_ptr<IHead> CompletePrediction::createHead() const {
    uint32 numElements = this->getNumElements();
    std::unique_ptr<CompleteHead> headPtr = std::make_unique<CompleteHead>(numElements);
    copyView(this->values_cbegin(), headPtr->values_begin(), numElements);
    return headPtr;
}
