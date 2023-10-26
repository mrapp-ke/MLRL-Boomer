#include "mlrl/common/rule_refinement/prediction_partial.hpp"

#include "mlrl/common/data/vector_sparse_array.hpp"
#include "mlrl/common/model/head_partial.hpp"
#include "mlrl/common/post_processing/post_processor.hpp"
#include "mlrl/common/rule_refinement/rule_refinement.hpp"
#include "mlrl/common/statistics/statistics.hpp"
#include "mlrl/common/util/view_functions.hpp"

PartialPrediction::PartialPrediction(uint32 numElements, bool sorted)
    : predictedScoreVector_(numElements), indexVector_(numElements), sorted_(sorted) {}

PartialPrediction::value_iterator PartialPrediction::values_begin() {
    return predictedScoreVector_.begin();
}

PartialPrediction::value_iterator PartialPrediction::values_end() {
    return predictedScoreVector_.end();
}

PartialPrediction::value_const_iterator PartialPrediction::values_cbegin() const {
    return predictedScoreVector_.cbegin();
}

PartialPrediction::value_const_iterator PartialPrediction::values_cend() const {
    return predictedScoreVector_.cend();
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
    return predictedScoreVector_.getNumElements();
}

void PartialPrediction::setNumElements(uint32 numElements, bool freeMemory) {
    this->predictedScoreVector_.setNumElements(numElements, freeMemory);
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

void PartialPrediction::set(DenseVector<float64>::const_iterator begin, DenseVector<float64>::const_iterator end) {
    copyArray(begin, predictedScoreVector_.begin(), predictedScoreVector_.getNumElements());
}

void PartialPrediction::set(DenseBinnedVector<float64>::const_iterator begin,
                            DenseBinnedVector<float64>::const_iterator end) {
    copyArray(begin, predictedScoreVector_.begin(), predictedScoreVector_.getNumElements());
}

bool PartialPrediction::isPartial() const {
    return true;
}

uint32 PartialPrediction::getIndex(uint32 pos) const {
    return indexVector_.getIndex(pos);
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

std::unique_ptr<IRuleRefinement> PartialPrediction::createRuleRefinement(IThresholdsSubset& thresholdsSubset,
                                                                         uint32 featureIndex) const {
    return indexVector_.createRuleRefinement(thresholdsSubset, featureIndex);
}

void PartialPrediction::apply(IStatistics& statistics, uint32 statisticIndex) const {
    statistics.applyPrediction(statisticIndex, *this);
}

void PartialPrediction::revert(IStatistics& statistics, uint32 statisticIndex) const {
    statistics.revertPrediction(statisticIndex, *this);
}

std::unique_ptr<IHead> PartialPrediction::createHead() const {
    uint32 numElements = this->getNumElements();
    std::unique_ptr<PartialHead> headPtr = std::make_unique<PartialHead>(numElements);
    copyView(this->values_cbegin(), headPtr->values_begin(), numElements);
    copyView(this->indices_cbegin(), headPtr->indices_begin(), numElements);
    return headPtr;
}
