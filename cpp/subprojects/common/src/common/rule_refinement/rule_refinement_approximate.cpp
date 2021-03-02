#include "common/rule_refinement/rule_refinement_approximate.hpp"
#include "rule_refinement_common.hpp"


template<class T>
ApproximateRuleRefinement<T>::ApproximateRuleRefinement(
        std::unique_ptr<IHeadRefinement> headRefinementPtr, const T& labelIndices, uint32 featureIndex, bool nominal,
        std::unique_ptr<IRuleRefinementCallback<BinVector, BinWeightVector>> callbackPtr)
    : headRefinementPtr_(std::move(headRefinementPtr)), labelIndices_(labelIndices), featureIndex_(featureIndex),
      nominal_(nominal), callbackPtr_(std::move(callbackPtr)) {

}

template<class T>
void ApproximateRuleRefinement<T>::findRefinement(const AbstractEvaluatedPrediction* currentHead) {
    std::unique_ptr<Refinement> refinementPtr = std::make_unique<Refinement>();
    refinementPtr->featureIndex = featureIndex_;
    refinementPtr->start = 0;
    const AbstractEvaluatedPrediction* bestHead = currentHead;

    // Invoke the callback...
    std::unique_ptr<IRuleRefinementCallback<BinVector, BinWeightVector>::Result> callbackResultPtr =
        callbackPtr_->get();
    const IImmutableStatistics& statistics = callbackResultPtr->statistics_;
    const BinWeightVector& weights = callbackResultPtr->weights_;
    BinWeightVector::const_iterator weightIterator = weights.cbegin();
    const BinVector& binVector = callbackResultPtr->vector_;
    BinVector::const_iterator binIterator = binVector.cbegin();
    uint32 numBins = binVector.getNumElements();

    // Create a new, empty subset of the current statistics when processing a new feature...
    std::unique_ptr<IStatisticsSubset> statisticsSubsetPtr = labelIndices_.createSubset(statistics);

    // Search for the first non-empty bin...
    uint32 r = 0;
    uint32 previousR = 0;
    float32 previousValue = 0;

    for (; r < numBins; r++) {
        uint32 weight = weightIterator[r];

        if (weight > 0) {
            previousValue = binIterator[r].maxValue;
            previousR = r;
            statisticsSubsetPtr->addToSubset(r, 1);
            break;
        }
    }

    for (r = r + 1; r < numBins; r++) {
        uint32 weight = weightIterator[r];

        if (weight > 0) {
            float32 currentValue = binIterator[r].minValue;

            const AbstractEvaluatedPrediction* head = headRefinementPtr_->findHead(bestHead, *statisticsSubsetPtr,
                                                                                   false, false);

            if (head != nullptr) {
                bestHead = head;
                refinementPtr->end = r;
                refinementPtr->previous = previousR;
                refinementPtr->covered = true;

                if (nominal_) {
                    refinementPtr->comparator = EQ;
                    refinementPtr->threshold = previousValue;
                } else {
                    refinementPtr->comparator = LEQ;
                    refinementPtr->threshold = calculateThreshold(previousValue, currentValue);
                }
            }

            head = headRefinementPtr_->findHead(bestHead, *statisticsSubsetPtr, true, false);

            if (head != nullptr) {
                bestHead = head;
                refinementPtr->end = r;
                refinementPtr->previous = previousR;
                refinementPtr->covered = false;

                if (nominal_) {
                    refinementPtr->comparator = NEQ;
                    refinementPtr->threshold = previousValue;
                } else {
                    refinementPtr->comparator = GR;
                    refinementPtr->threshold = calculateThreshold(previousValue, currentValue);
                }
            }

            previousValue = binIterator[r].maxValue;
            previousR = r;
            statisticsSubsetPtr->addToSubset(r, 1);

            // Reset the subset in case of a nominal feature, as the previous bins will not be covered by the next
            // condition...
            if (nominal_) {
                statisticsSubsetPtr->resetSubset();
            }
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
