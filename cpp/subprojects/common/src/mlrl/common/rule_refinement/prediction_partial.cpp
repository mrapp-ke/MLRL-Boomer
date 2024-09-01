#include "mlrl/common/rule_refinement/prediction_partial.hpp"

#include "mlrl/common/data/vector_sparse_array.hpp"
#include "mlrl/common/model/head_partial.hpp"
#include "mlrl/common/post_processing/post_processor.hpp"
#include "mlrl/common/rule_refinement/rule_refinement.hpp"
#include "mlrl/common/statistics/statistics.hpp"

PartialPrediction::PartialPrediction(uint32 numElements, bool sorted)
    : ResizableVectorDecorator<VectorDecorator<ResizableVector<float64>>>(ResizableVector<float64>(numElements)),
      indexVector_(numElements), sorted_(sorted) {}

PartialPrediction::value_iterator PartialPrediction::values_begin() {
    return this->view.begin();
}

PartialPrediction::value_iterator PartialPrediction::values_end() {
    return this->view.end();
}

PartialPrediction::value_const_iterator PartialPrediction::values_cbegin() const {
    return this->view.cbegin();
}

PartialPrediction::value_const_iterator PartialPrediction::values_cend() const {
    return this->view.cend();
}

PartialPrediction::index_iterator PartialPrediction::indices_begin() {
    return indexVector_.begin();
}

PartialPrediction::index_iterator PartialPrediction::indices_end() {
    return indexVector_.end();
}

PartialPrediction::index_const_iterator PartialPrediction::indices_cbegin() const {
    return indexVector_.cbegin();
}

PartialPrediction::index_const_iterator PartialPrediction::indices_cend() const {
    return indexVector_.cend();
}

uint32 PartialPrediction::getNumElements() const {
    return ResizableVectorDecorator<VectorDecorator<ResizableVector<float64>>>::getNumElements();
}

void PartialPrediction::setNumElements(uint32 numElements, bool freeMemory) {
    ResizableVectorDecorator<VectorDecorator<ResizableVector<float64>>>::setNumElements(numElements, freeMemory);
    indexVector_.setNumElements(numElements, freeMemory);
}

void PartialPrediction::setSorted(bool sorted) {
    sorted_ = sorted;
}

void PartialPrediction::sort() {
    if (!sorted_) {
        uint32 numElements = this->getNumElements();

        if (numElements > 1) {
            SparseArrayVector<float64> sortedVector(numElements);
            SparseArrayVector<float64>::iterator sortedIterator = sortedVector.begin();
            index_iterator indexIterator = this->indices_begin();
            value_iterator valueIterator = this->values_begin();

            for (uint32 i = 0; i < numElements; i++) {
                IndexedValue<float64>& entry = sortedIterator[i];
                entry.index = indexIterator[i];
                entry.value = valueIterator[i];
            }

            std::sort(sortedIterator, sortedVector.end(), IndexedValue<float64>::CompareIndex());

            for (uint32 i = 0; i < numElements; i++) {
                const IndexedValue<float64>& entry = sortedIterator[i];
                indexIterator[i] = entry.index;
                valueIterator[i] = entry.value;
            }
        }

        sorted_ = true;
    }
}

void PartialPrediction::postProcess(const IPostProcessor& postProcessor) {
    postProcessor.postProcess(this->values_begin(), this->values_end());
}

void PartialPrediction::set(View<float64>::const_iterator begin, View<float64>::const_iterator end) {
    util::copyView(begin, this->view.begin(), this->getNumElements());
}

void PartialPrediction::set(BinnedConstIterator<float64> begin, BinnedConstIterator<float64> end) {
    util::copyView(begin, this->view.begin(), this->getNumElements());
}

bool PartialPrediction::isPartial() const {
    return true;
}

uint32 PartialPrediction::getIndex(uint32 pos) const {
    return indexVector_.getIndex(pos);
}

void PartialPrediction::visit(PartialIndexVectorVisitor partialIndexVectorVisitor,
                              CompleteIndexVectorVisitor completeIndexVectorVisitor) const {
    partialIndexVectorVisitor(indexVector_);
}

std::unique_ptr<IStatisticsSubset> PartialPrediction::createStatisticsSubset(const IStatistics& statistics,
                                                                             const EqualWeightVector& weights) const {
    return statistics.createSubset(indexVector_, weights);
}

std::unique_ptr<IStatisticsSubset> PartialPrediction::createStatisticsSubset(const IStatistics& statistics,
                                                                             const BitWeightVector& weights) const {
    return statistics.createSubset(indexVector_, weights);
}

std::unique_ptr<IStatisticsSubset> PartialPrediction::createStatisticsSubset(
  const IStatistics& statistics, const DenseWeightVector<uint32>& weights) const {
    return statistics.createSubset(indexVector_, weights);
}

std::unique_ptr<IStatisticsSubset> PartialPrediction::createStatisticsSubset(
  const IStatistics& statistics, const OutOfSampleWeightVector<EqualWeightVector>& weights) const {
    return statistics.createSubset(indexVector_, weights);
}

std::unique_ptr<IStatisticsSubset> PartialPrediction::createStatisticsSubset(
  const IStatistics& statistics, const OutOfSampleWeightVector<BitWeightVector>& weights) const {
    return statistics.createSubset(indexVector_, weights);
}

std::unique_ptr<IStatisticsSubset> PartialPrediction::createStatisticsSubset(
  const IStatistics& statistics, const OutOfSampleWeightVector<DenseWeightVector<uint32>>& weights) const {
    return statistics.createSubset(indexVector_, weights);
}

std::unique_ptr<IRuleRefinement> PartialPrediction::createRuleRefinement(IFeatureSubspace& featureSubspace,
                                                                         uint32 featureIndex,
                                                                         const IWeightedStatistics& statistics,
                                                                         const IFeatureVector& featureVector) const {
    return indexVector_.createRuleRefinement(featureSubspace, featureIndex, statistics, featureVector);
}

std::unique_ptr<IStatisticsUpdate> PartialPrediction::createStatisticsUpdate(IStatistics& statistics) const {
    return statistics.createUpdate(*this);
}

std::unique_ptr<IHead> PartialPrediction::createHead() const {
    uint32 numElements = this->getNumElements();
    std::unique_ptr<PartialHead> headPtr = std::make_unique<PartialHead>(numElements);
    util::copyView(this->values_cbegin(), headPtr->values_begin(), numElements);
    util::copyView(this->indices_cbegin(), headPtr->indices_begin(), numElements);
    return headPtr;
}
