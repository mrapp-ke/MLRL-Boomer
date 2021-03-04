#include "common/rule_refinement/rule_refinement_approximate.hpp"


template<class T>
ApproximateRuleRefinement<T>::ApproximateRuleRefinement(
        std::unique_ptr<IHeadRefinement> headRefinementPtr, const T& labelIndices, uint32 totalSumOfWeights,
        uint32 featureIndex, bool nominal, const IWeightVector& weights,
        std::unique_ptr<IRuleRefinementCallback<ThresholdVector, BinWeightVector>> callbackPtr)
    : headRefinementPtr_(std::move(headRefinementPtr)), labelIndices_(labelIndices),
      totalSumOfWeights_(totalSumOfWeights), featureIndex_(featureIndex), nominal_(nominal), weights_(weights),
      callbackPtr_(std::move(callbackPtr)) {

}

template<class T>
void ApproximateRuleRefinement<T>::findRefinement(const AbstractEvaluatedPrediction* currentHead) {
    std::unique_ptr<Refinement> refinementPtr = std::make_unique<Refinement>();
    refinementPtr->featureIndex = featureIndex_;
    refinementPtr->start = 0;
    const AbstractEvaluatedPrediction* bestHead = currentHead;

    // Invoke the callback...
    std::unique_ptr<IRuleRefinementCallback<ThresholdVector, BinWeightVector>::Result> callbackResultPtr =
        callbackPtr_->get();
    const IImmutableStatistics& statistics = callbackResultPtr->statistics_;
    const BinWeightVector& weights = callbackResultPtr->weights_;
    BinWeightVector::const_iterator weightIterator = weights.cbegin();
    const ThresholdVector& thresholdVector = callbackResultPtr->vector_;
    ThresholdVector::const_iterator thresholdIterator = thresholdVector.cbegin();
    uint32 numBins = thresholdVector.getNumElements();

    // Create a new, empty subset of the statistics...
    std::unique_ptr<IStatisticsSubset> statisticsSubsetPtr = labelIndices_.createSubset(statistics);

    for (auto it = thresholdVector.missing_indices_cbegin(); it != thresholdVector.missing_indices_cend(); it++) {
        uint32 i = *it;
        uint32 weight = weights_.getWeight(i);
        statisticsSubsetPtr->addToMissing(i, weight);
    }

    // Traverse bins in ascending order until the first bin with weight > 0 is encountered...
    uint32 r = 0;
    float32 threshold = 0;
    uint32 sumOfWeights = 0;

    for (; r < numBins; r++) {
        uint32 weight = weightIterator[r];

        if (weight > 0) {
            // Add the bin to the subset to mark it as covered by upcoming refinements...
            threshold = thresholdIterator[r];
            statisticsSubsetPtr->addToSubset(r, 1);
            sumOfWeights += weight;
            break;
        }
    }

    // Traverse the remaining bins in ascending order...
    for (r = r + 1; r < numBins; r++) {
        uint32 weight = weightIterator[r];

        // Do only consider bins that are not empty...
        if (weight > 0) {
            // Find and evaluate the best head for the current refinement, if a condition that uses the <= operator (or
            // the == operator in case of a nominal feature) is used...
            const AbstractEvaluatedPrediction* head = headRefinementPtr_->findHead(bestHead, *statisticsSubsetPtr,
                                                                                   false, false);

            // If the refinement is better than the current rule...
            if (head != nullptr) {
                bestHead = head;
                refinementPtr->end = r;
                refinementPtr->coveredWeights = sumOfWeights;
                refinementPtr->covered = true;
                refinementPtr->threshold = threshold;
                refinementPtr->comparator = nominal_ ? EQ : LEQ;
            }

            // Find and evaluate the best head for the current refinement, if a condition that uses the > operator (or
            // the != operator in case of a nominal feature) is used...
            head = headRefinementPtr_->findHead(bestHead, *statisticsSubsetPtr, true, false);

            // If the refinement is better than the current rule...
            if (head != nullptr) {
                bestHead = head;
                refinementPtr->end = r;
                refinementPtr->coveredWeights = (totalSumOfWeights_ - sumOfWeights);
                refinementPtr->covered = false;
                refinementPtr->threshold = threshold;
                refinementPtr->comparator = nominal_ ? NEQ : GR;
            }

            // Reset the subset in case of a nominal feature, as the previous bins will not be covered by the next
            // condition...
            if (nominal_) {
                statisticsSubsetPtr->resetSubset();
                sumOfWeights = 0;
            }

            threshold = thresholdIterator[r];

            // Add the bin to the subset to mark it as covered by upcoming refinements...
            statisticsSubsetPtr->addToSubset(r, 1);
            sumOfWeights += weight;
        }
    }

    refinementPtr->headPtr = headRefinementPtr_->pollHead();
    refinementPtr_ = std::move(refinementPtr);
}

template<class T>
std::unique_ptr<Refinement> ApproximateRuleRefinement<T>::pollRefinement() {
    return std::move(refinementPtr_);
}

template class ApproximateRuleRefinement<FullIndexVector>;
template class ApproximateRuleRefinement<PartialIndexVector>;
