#include "rule_refinement_approximate.h"

template<class T>
ApproximateRuleRefinement<T>::ApproximateRuleRefinement(std::unique_ptr<IHeadRefinement> headRefinementPtr,
                                                        const T& labelIndices, uint32 featureIndex,
                                                        std::unique_ptr<IRuleRefinementCallback<BinVector>> callbackPtr)
    : headRefinementPtr_(std::move(headRefinementPtr)), labelIndices_(labelIndices), featureIndex_(featureIndex),
      callbackPtr_(std::move(callbackPtr)) {

}

template<class T>
void ApproximateRuleRefinement<T>::findRefinement(const AbstractEvaluatedPrediction* currentHead) {
    std::unique_ptr<Refinement> refinementPtr = std::make_unique<Refinement>();
    refinementPtr->featureIndex = featureIndex_;
    refinementPtr->start = 0;
    const AbstractEvaluatedPrediction* bestHead = currentHead;

    // Invoke the callback...
    std::unique_ptr<IRuleRefinementCallback<BinVector>::Result> callbackResultPtr = callbackPtr_->get();
    const IHistogram& histogram = callbackResultPtr->first;
    const BinVector& binVector = callbackResultPtr->second;
    BinVector::const_iterator iterator = binVector.cbegin();
    uint32 numBins = binVector.getNumElements();

    // Create a new, empty subset of the current statistics when processing a new feature...
    std::unique_ptr<IStatisticsSubset> statisticsSubsetPtr = labelIndices_.createSubset(histogram);

    // Search for the first non-empty bin...
    uint32 r = 0;
    uint32 previousR = 0;
    float32 previousValue = 0;
    uint32 numCoveredExamples = 0;

    for (; r < numBins; r++) {
        if (iterator[r].numExamples > 0){
            statisticsSubsetPtr->addToSubset(r, 1);
            previousR = r;
            previousValue = iterator[r].maxValue;
            numCoveredExamples = iterator[r].numExamples;
            break;
        }
    }
    if (numCoveredExamples > 0){
        for (r += 1; r < numBins; r++) {
            uint32 numExamples = iterator[r].numExamples;

            if (numExamples > 0) {
                float32 currentValue = iterator[r].minValue;

                const AbstractEvaluatedPrediction* head = headRefinementPtr_->findHead(bestHead, *statisticsSubsetPtr,
                                                                                   false, false);
                if (head != nullptr) {
                    bestHead = head;
                    refinementPtr->comparator = LEQ;
                    refinementPtr->threshold = (previousValue + currentValue) / 2.0;
                    refinementPtr->end = r;
                    refinementPtr->previous = previousR;
                    refinementPtr->coveredWeights = numCoveredExamples;
                    refinementPtr->covered = true;
                }

                head = headRefinementPtr_->findHead(bestHead, *statisticsSubsetPtr, true, false);

                if (head != nullptr) {
                    bestHead = head;
                    refinementPtr->comparator = GR;
                    refinementPtr->threshold = (previousValue + currentValue) / 2.0;
                    refinementPtr->end = r;
                    refinementPtr->previous = previousR;
                    refinementPtr->coveredWeights = numCoveredExamples;
                    refinementPtr->covered = false;
                }

                previousValue = iterator[r].maxValue;
                numCoveredExamples += numExamples;
                statisticsSubsetPtr->addToSubset(r, 1);
                previousR = r;
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
