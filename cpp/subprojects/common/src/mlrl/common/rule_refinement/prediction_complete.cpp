#include "mlrl/common/rule_refinement/prediction_complete.hpp"

#include "mlrl/common/model/head_complete.hpp"
#include "mlrl/common/post_processing/post_processor.hpp"
#include "mlrl/common/rule_refinement/rule_refinement.hpp"
#include "mlrl/common/statistics/statistics.hpp"

static inline std::unique_ptr<IHead> createHeadInternally(const CompletePrediction<uint8>& prediction) {
    uint32 numElements = prediction.getNumElements();
    std::unique_ptr<CompleteHead<float32>> headPtr = std::make_unique<CompleteHead<float32>>(numElements);
    util::copyView(prediction.values_cbegin(), headPtr->values_begin(), numElements);
    return headPtr;
}

template<typename ScoreType>
static inline std::unique_ptr<IHead> createHeadInternally(const CompletePrediction<ScoreType>& prediction) {
    uint32 numElements = prediction.getNumElements();
    std::unique_ptr<CompleteHead<ScoreType>> headPtr = std::make_unique<CompleteHead<ScoreType>>(numElements);
    util::copyView(prediction.values_cbegin(), headPtr->values_begin(), numElements);
    return headPtr;
}

template<typename ScoreType>
CompletePrediction<ScoreType>::CompletePrediction(uint32 numElements,
                                                  IStatisticsUpdateFactory<ScoreType>& statisticsUpdateFactory)
    : VectorDecorator<AllocatedVector<ScoreType>>(AllocatedVector<ScoreType>(numElements)), indexVector_(numElements),
      statisticsUpdatePtr_(statisticsUpdateFactory.create(indexVector_.cbegin(), indexVector_.cend(),
                                                          this->view.cbegin(), this->view.cend())) {}

template<typename ScoreType>
typename CompletePrediction<ScoreType>::value_iterator CompletePrediction<ScoreType>::values_begin() {
    return this->view.begin();
}

template<typename ScoreType>
typename CompletePrediction<ScoreType>::value_iterator CompletePrediction<ScoreType>::values_end() {
    return this->view.end();
}

template<typename ScoreType>
typename CompletePrediction<ScoreType>::value_const_iterator CompletePrediction<ScoreType>::values_cbegin() const {
    return this->view.cbegin();
}

template<typename ScoreType>
typename CompletePrediction<ScoreType>::value_const_iterator CompletePrediction<ScoreType>::values_cend() const {
    return this->view.cend();
}

template<typename ScoreType>
typename CompletePrediction<ScoreType>::index_const_iterator CompletePrediction<ScoreType>::indices_cbegin() const {
    return indexVector_.cbegin();
}

template<typename ScoreType>
typename CompletePrediction<ScoreType>::index_const_iterator CompletePrediction<ScoreType>::indices_cend() const {
    return indexVector_.cend();
}

template<typename ScoreType>
uint32 CompletePrediction<ScoreType>::getNumElements() const {
    return VectorDecorator<AllocatedVector<ScoreType>>::getNumElements();
}

template<typename ScoreType>
void CompletePrediction<ScoreType>::sort() {}

template<typename ScoreType>
void CompletePrediction<ScoreType>::postProcess(const IPostProcessor& postProcessor) {
    postProcessor.postProcess(this->values_begin(), this->values_end());
}

template<typename ScoreType>
bool CompletePrediction<ScoreType>::isPartial() const {
    return false;
}

template<typename ScoreType>
uint32 CompletePrediction<ScoreType>::getIndex(uint32 pos) const {
    return indexVector_.getIndex(pos);
}

template<typename ScoreType>
void CompletePrediction<ScoreType>::visit(PartialIndexVectorVisitor partialIndexVectorVisitor,
                                          CompleteIndexVectorVisitor completeIndexVectorVisitor) const {
    completeIndexVectorVisitor(indexVector_);
}

template<typename ScoreType>
std::unique_ptr<IStatisticsSubset> CompletePrediction<ScoreType>::createStatisticsSubset(
  const IStatistics& statistics, const EqualWeightVector& weights) const {
    return statistics.createSubset(indexVector_, weights);
}

template<typename ScoreType>
std::unique_ptr<IStatisticsSubset> CompletePrediction<ScoreType>::createStatisticsSubset(
  const IStatistics& statistics, const BitWeightVector& weights) const {
    return statistics.createSubset(indexVector_, weights);
}

template<typename ScoreType>
std::unique_ptr<IStatisticsSubset> CompletePrediction<ScoreType>::createStatisticsSubset(
  const IStatistics& statistics, const DenseWeightVector<uint16>& weights) const {
    return statistics.createSubset(indexVector_, weights);
}

template<typename ScoreType>
std::unique_ptr<IStatisticsSubset> CompletePrediction<ScoreType>::createStatisticsSubset(
  const IStatistics& statistics, const DenseWeightVector<float32>& weights) const {
    return statistics.createSubset(indexVector_, weights);
}

template<typename ScoreType>
std::unique_ptr<IStatisticsSubset> CompletePrediction<ScoreType>::createStatisticsSubset(
  const IStatistics& statistics, const OutOfSampleWeightVector<EqualWeightVector>& weights) const {
    return statistics.createSubset(indexVector_, weights);
}

template<typename ScoreType>
std::unique_ptr<IStatisticsSubset> CompletePrediction<ScoreType>::createStatisticsSubset(
  const IStatistics& statistics, const OutOfSampleWeightVector<BitWeightVector>& weights) const {
    return statistics.createSubset(indexVector_, weights);
}

template<typename ScoreType>
std::unique_ptr<IStatisticsSubset> CompletePrediction<ScoreType>::createStatisticsSubset(
  const IStatistics& statistics, const OutOfSampleWeightVector<DenseWeightVector<uint16>>& weights) const {
    return statistics.createSubset(indexVector_, weights);
}

template<typename ScoreType>
std::unique_ptr<IStatisticsSubset> CompletePrediction<ScoreType>::createStatisticsSubset(
  const IStatistics& statistics, const OutOfSampleWeightVector<DenseWeightVector<float32>>& weights) const {
    return statistics.createSubset(indexVector_, weights);
}

template<typename ScoreType>
void CompletePrediction<ScoreType>::applyPrediction(uint32 statisticIndex) {
    statisticsUpdatePtr_->applyPrediction(statisticIndex);
}

template<typename ScoreType>
void CompletePrediction<ScoreType>::revertPrediction(uint32 statisticIndex) {
    statisticsUpdatePtr_->revertPrediction(statisticIndex);
}

template<typename ScoreType>
std::unique_ptr<IHead> CompletePrediction<ScoreType>::createHead() const {
    return createHeadInternally(*this);
}

template class CompletePrediction<uint8>;
template class CompletePrediction<float32>;
template class CompletePrediction<float64>;
