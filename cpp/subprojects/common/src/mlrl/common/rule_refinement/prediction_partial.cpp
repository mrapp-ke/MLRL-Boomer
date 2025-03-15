#include "mlrl/common/rule_refinement/prediction_partial.hpp"

#include "mlrl/common/data/vector_sparse_array.hpp"
#include "mlrl/common/model/head_partial.hpp"
#include "mlrl/common/post_processing/post_processor.hpp"
#include "mlrl/common/rule_refinement/rule_refinement.hpp"
#include "mlrl/common/statistics/statistics.hpp"

static inline std::unique_ptr<IHead> createHeadInternally(const PartialPrediction<uint8>& prediction) {
    uint32 numElements = prediction.getNumElements();
    std::unique_ptr<PartialHead<float32>> headPtr = std::make_unique<PartialHead<float32>>(numElements);
    util::copyView(prediction.values_cbegin(), headPtr->values_begin(), numElements);
    util::copyView(prediction.indices_cbegin(), headPtr->indices_begin(), numElements);
    return headPtr;
}

template<typename ScoreType>
static inline std::unique_ptr<IHead> createHeadInternally(const PartialPrediction<ScoreType>& prediction) {
    uint32 numElements = prediction.getNumElements();
    std::unique_ptr<PartialHead<ScoreType>> headPtr = std::make_unique<PartialHead<ScoreType>>(numElements);
    util::copyView(prediction.values_cbegin(), headPtr->values_begin(), numElements);
    util::copyView(prediction.indices_cbegin(), headPtr->indices_begin(), numElements);
    return headPtr;
}

template<typename ScoreType>
PartialPrediction<ScoreType>::PartialPrediction(uint32 numElements, bool sorted,
                                                IStatisticsUpdateFactory<ScoreType>& statisticsUpdateFactory)
    : VectorDecorator<ResizableVector<ScoreType>>(ResizableVector<ScoreType>(numElements)), indexVector_(numElements),
      sorted_(sorted), statisticsUpdatePtr_(statisticsUpdateFactory.create(indexVector_.cbegin(), indexVector_.cend(),
                                                                           this->view.cbegin(), this->view.cend())) {}

template<typename ScoreType>
typename PartialPrediction<ScoreType>::value_iterator PartialPrediction<ScoreType>::values_begin() {
    return this->view.begin();
}

template<typename ScoreType>
typename PartialPrediction<ScoreType>::value_iterator PartialPrediction<ScoreType>::values_end() {
    return this->view.end();
}

template<typename ScoreType>
typename PartialPrediction<ScoreType>::value_const_iterator PartialPrediction<ScoreType>::values_cbegin() const {
    return this->view.cbegin();
}

template<typename ScoreType>
typename PartialPrediction<ScoreType>::value_const_iterator PartialPrediction<ScoreType>::values_cend() const {
    return this->view.cend();
}

template<typename ScoreType>
typename PartialPrediction<ScoreType>::index_iterator PartialPrediction<ScoreType>::indices_begin() {
    return indexVector_.begin();
}

template<typename ScoreType>
typename PartialPrediction<ScoreType>::index_iterator PartialPrediction<ScoreType>::indices_end() {
    return indexVector_.end();
}

template<typename ScoreType>
typename PartialPrediction<ScoreType>::index_const_iterator PartialPrediction<ScoreType>::indices_cbegin() const {
    return indexVector_.cbegin();
}

template<typename ScoreType>
typename PartialPrediction<ScoreType>::index_const_iterator PartialPrediction<ScoreType>::indices_cend() const {
    return indexVector_.cend();
}

template<typename ScoreType>
uint32 PartialPrediction<ScoreType>::getNumElements() const {
    return VectorDecorator<ResizableVector<ScoreType>>::getNumElements();
}

template<typename ScoreType>
void PartialPrediction<ScoreType>::setNumElements(IStatisticsUpdateFactory<ScoreType>& statisticsUpdateFactory,
                                                  uint32 numElements, bool freeMemory) {
    this->view.resize(numElements, freeMemory);
    indexVector_.setNumElements(numElements, freeMemory);
    statisticsUpdatePtr_ = statisticsUpdateFactory.create(this->indices_cbegin(), this->indices_cend(),
                                                          this->values_cbegin(), this->values_cend());
}

template<typename ScoreType>
void PartialPrediction<ScoreType>::setSorted(bool sorted) {
    sorted_ = sorted;
}

template<typename ScoreType>
void PartialPrediction<ScoreType>::sort() {
    if (!sorted_) {
        uint32 numElements = this->getNumElements();

        if (numElements > 1) {
            SparseArrayVector<ScoreType> sortedVector(numElements);
            typename SparseArrayVector<ScoreType>::iterator sortedIterator = sortedVector.begin();
            index_iterator indexIterator = this->indices_begin();
            value_iterator valueIterator = this->values_begin();

            for (uint32 i = 0; i < numElements; i++) {
                IndexedValue<ScoreType>& entry = sortedIterator[i];
                entry.index = indexIterator[i];
                entry.value = valueIterator[i];
            }

            std::sort(sortedIterator, sortedVector.end(), typename IndexedValue<ScoreType>::CompareIndex());

            for (uint32 i = 0; i < numElements; i++) {
                const IndexedValue<ScoreType>& entry = sortedIterator[i];
                indexIterator[i] = entry.index;
                valueIterator[i] = entry.value;
            }
        }

        sorted_ = true;
    }
}

template<typename ScoreType>
void PartialPrediction<ScoreType>::postProcess(const IPostProcessor& postProcessor) {
    postProcessor.postProcess(this->values_begin(), this->values_end());
}

template<typename ScoreType>
bool PartialPrediction<ScoreType>::isPartial() const {
    return true;
}

template<typename ScoreType>
uint32 PartialPrediction<ScoreType>::getIndex(uint32 pos) const {
    return indexVector_.getIndex(pos);
}

template<typename ScoreType>
void PartialPrediction<ScoreType>::visit(PartialIndexVectorVisitor partialIndexVectorVisitor,
                                         CompleteIndexVectorVisitor completeIndexVectorVisitor) const {
    partialIndexVectorVisitor(indexVector_);
}

template<typename ScoreType>
std::unique_ptr<IStatisticsSubset> PartialPrediction<ScoreType>::createStatisticsSubset(
  const IStatistics& statistics, const EqualWeightVector& weights) const {
    return statistics.createSubset(indexVector_, weights);
}

template<typename ScoreType>
std::unique_ptr<IStatisticsSubset> PartialPrediction<ScoreType>::createStatisticsSubset(
  const IStatistics& statistics, const BitWeightVector& weights) const {
    return statistics.createSubset(indexVector_, weights);
}

template<typename ScoreType>
std::unique_ptr<IStatisticsSubset> PartialPrediction<ScoreType>::createStatisticsSubset(
  const IStatistics& statistics, const DenseWeightVector<uint16>& weights) const {
    return statistics.createSubset(indexVector_, weights);
}

template<typename ScoreType>
std::unique_ptr<IStatisticsSubset> PartialPrediction<ScoreType>::createStatisticsSubset(
  const IStatistics& statistics, const DenseWeightVector<float32>& weights) const {
    return statistics.createSubset(indexVector_, weights);
}

template<typename ScoreType>
std::unique_ptr<IStatisticsSubset> PartialPrediction<ScoreType>::createStatisticsSubset(
  const IStatistics& statistics, const OutOfSampleWeightVector<EqualWeightVector>& weights) const {
    return statistics.createSubset(indexVector_, weights);
}

template<typename ScoreType>
std::unique_ptr<IStatisticsSubset> PartialPrediction<ScoreType>::createStatisticsSubset(
  const IStatistics& statistics, const OutOfSampleWeightVector<BitWeightVector>& weights) const {
    return statistics.createSubset(indexVector_, weights);
}

template<typename ScoreType>
std::unique_ptr<IStatisticsSubset> PartialPrediction<ScoreType>::createStatisticsSubset(
  const IStatistics& statistics, const OutOfSampleWeightVector<DenseWeightVector<uint16>>& weights) const {
    return statistics.createSubset(indexVector_, weights);
}

template<typename ScoreType>
std::unique_ptr<IStatisticsSubset> PartialPrediction<ScoreType>::createStatisticsSubset(
  const IStatistics& statistics, const OutOfSampleWeightVector<DenseWeightVector<float32>>& weights) const {
    return statistics.createSubset(indexVector_, weights);
}

template<typename ScoreType>
void PartialPrediction<ScoreType>::applyPrediction(uint32 statisticIndex) {
    statisticsUpdatePtr_->applyPrediction(statisticIndex);
}

template<typename ScoreType>
void PartialPrediction<ScoreType>::revertPrediction(uint32 statisticIndex) {
    statisticsUpdatePtr_->revertPrediction(statisticIndex);
}

template<typename ScoreType>
std::unique_ptr<IHead> PartialPrediction<ScoreType>::createHead() const {
    return createHeadInternally(*this);
}

template class PartialPrediction<uint8>;
template class PartialPrediction<float32>;
template class PartialPrediction<float64>;
