#include "rule_refinement_approximate.h"
#include "rule_refinement_common.h"


template<class T>
ApproximateRuleRefinement<T>::ApproximateRuleRefinement(
        std::unique_ptr<IHeadRefinement> headRefinementPtr, const T& labelIndices, uint32 featureIndex, bool nominal,
        std::unique_ptr<IRuleRefinementCallback<BinVector, DenseVector<uint32>>> callbackPtr)
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
    std::unique_ptr<IRuleRefinementCallback<BinVector, DenseVector<uint32>>::Result> callbackResultPtr =
        callbackPtr_->get();
    const IImmutableStatistics& statistics = callbackResultPtr->statistics_;
    const DenseVector<uint32>& weights = callbackResultPtr->weights_;
    DenseVector<uint32>::const_iterator weightIterator = weights.cbegin();
    const BinVector& binVector = callbackResultPtr->vector_;
    BinVector::bin_const_iterator binIterator = binVector.bins_cbegin();
    uint32 numBins = binVector.getNumElements();

    // Create a new, empty subset of the current statistics when processing a new feature...
    std::unique_ptr<IStatisticsSubset> statisticsSubsetPtr = labelIndices_.createSubset(statistics);

    // Search for the first non-empty bin...
    uint32 r = 0;
    uint32 previousR = 0;
    float32 previousValue = 0;
    uint32 sumOfWeights = 0;

    for (; r < numBins; r++) {
        uint32 binIndex = binIterator[r].index;
        uint32 weight = weightIterator[binIndex];

        if (weight > 0) {
            previousValue = binIterator[r].maxValue;
            previousR = r;
            sumOfWeights += weight;
            statisticsSubsetPtr->addToSubset(binIndex, 1);
            break;
        }
    }

    if (sumOfWeights > 0) {
        for (r = r + 1; r < numBins; r++) {
            uint32 binIndex = binIterator[r].index;
            uint32 weight = weightIterator[binIndex];

            if (weight > 0) {
                float32 currentValue = binIterator[r].minValue;

                const AbstractEvaluatedPrediction* head = headRefinementPtr_->findHead(bestHead, *statisticsSubsetPtr,
                                                                                       false, false);

                if (head != nullptr) {
                    bestHead = head;
                    refinementPtr->end = r;
                    refinementPtr->previous = previousR;
                    refinementPtr->coveredWeights = sumOfWeights;
                    refinementPtr->covered = true;

                    if (nominal_) {
                        refinementPtr->comparator = LEQ;
                        refinementPtr->threshold = calculateThreshold(previousValue, currentValue);
                    } else {
                        refinementPtr->comparator = EQ;
                        refinementPtr->threshold = previousValue;
                    }
                }

                head = headRefinementPtr_->findHead(bestHead, *statisticsSubsetPtr, true, false);

                if (head != nullptr) {
                    bestHead = head;
                    refinementPtr->end = r;
                    refinementPtr->previous = previousR;
                    refinementPtr->coveredWeights = sumOfWeights;
                    refinementPtr->covered = false;

                    if (nominal_) {
                        refinementPtr->comparator = GR;
                        refinementPtr->threshold = calculateThreshold(previousValue, currentValue);
                    } else {
                        refinementPtr->comparator = NEQ;
                        refinementPtr->threshold = previousValue;
                    }
                }

                previousValue = binIterator[r].maxValue;
                previousR = r;
                sumOfWeights += weight;
                statisticsSubsetPtr->addToSubset(binIndex, 1);

                // Reset the subset in case of a nominal feature, as the previous bins will not be covered by the next
                // condition...
                if (nominal_) {
                    statisticsSubsetPtr->resetSubset();
                    sumOfWeights = 0;
                }
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
