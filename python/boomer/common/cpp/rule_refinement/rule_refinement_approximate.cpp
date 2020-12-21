#include "rule_refinement_approximate.h"


template<class T>
ApproximateRuleRefinement<T>::ApproximateRuleRefinement(
        std::unique_ptr<IHeadRefinement> headRefinementPtr, const T& labelIndices, uint32 featureIndex,
        std::unique_ptr<IRuleRefinementCallback<BinVector, DenseVector<uint32>>> callbackPtr)
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
    std::unique_ptr<IRuleRefinementCallback<BinVector, DenseVector<uint32>>::Result> callbackResultPtr =
        callbackPtr_->get();
    const IImmutableStatistics& statistics = callbackResultPtr->statistics_;
    const BinVector& binVector = callbackResultPtr->vector_;
    BinVector::bin_const_iterator binIterator = binVector.bins_cbegin();
    uint32 numBins = binVector.getNumElements();

    // Create a new, empty subset of the current statistics when processing a new feature...
    std::unique_ptr<IStatisticsSubset> statisticsSubsetPtr = labelIndices_.createSubset(statistics);

    // Search for the first non-empty bin...
    uint32 r = 0;
    uint32 previousR = 0;
    float32 previousValue = 0;
    uint32 numCoveredExamples = 0;

    for (; r < numBins; r++) {
        uint32 numExamples = binIterator[r].numExamples;
        uint32 binIndex = binIterator[r].index;

        if (numExamples > 0) {
            previousValue = binIterator[r].maxValue;
            previousR = r;
            numCoveredExamples += numExamples;
            statisticsSubsetPtr->addToSubset(binIndex, 1);
            break;
        }
    }

    if (numCoveredExamples > 0) {
        for (r = r + 1; r < numBins; r++) {
            uint32 numExamples = binIterator[r].numExamples;

            if (numExamples > 0) {
                float32 currentValue = binIterator[r].minValue;
                uint32 binIndex = binIterator[r].index;

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

                previousValue = binIterator[r].maxValue;
                previousR = r;
                numCoveredExamples += numExamples;
                statisticsSubsetPtr->addToSubset(binIndex, 1);
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
