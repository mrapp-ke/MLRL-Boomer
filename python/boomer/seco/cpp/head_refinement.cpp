#include "head_refinement.h"
#include <cstdlib>

using namespace seco;


static inline uint32* argsort(const float64* a, uint32 numElements) {
    IndexedValue<float64> tmpArray[numElements];

    for (uint32 i = 0; i < numElements; i++) {
        tmpArray[i].index = i;
        tmpArray[i].value = a[i];
    }

    qsort(&tmpArray, numElements, sizeof(IndexedValue<float64>), &tuples::compareIndexedValue<float64>);
    uint32* sortedArray = new uint32[numElements];

    for (uint32 i = 0; i < numElements; i++) {
        sortedArray[i] = tmpArray[i].index;
    }

    return sortedArray;
}

template<class T>
PartialHeadRefinementImpl<T>::PartialHeadRefinementImpl(const T& labelIndices,
                                                        std::shared_ptr<ILiftFunction> liftFunctionPtr)
    : labelIndices_(labelIndices), liftFunctionPtr_(liftFunctionPtr) {

}

template<class T>
const AbstractEvaluatedPrediction* PartialHeadRefinementImpl<T>::findHead(const AbstractEvaluatedPrediction* bestHead,
                                                                          IStatisticsSubset& statisticsSubset,
                                                                          bool uncovered, bool accumulated) {
    const AbstractEvaluatedPrediction* result = nullptr;
    const LabelWiseEvaluatedPrediction& prediction = statisticsSubset.calculateLabelWisePrediction(uncovered,
                                                                                                   accumulated);
    uint32 numPredictions = prediction.getNumElements();
    LabelWiseEvaluatedPrediction::const_iterator valueIterator = prediction.cbegin();
    LabelWiseEvaluatedPrediction::quality_score_const_iterator qualityScoreIterator =
        prediction.quality_scores_cbegin();
    uint32* sortedIndices = nullptr;
    float64 sumOfQualityScores = 0;
    uint32 bestNumPredictions = 0;
    float64 bestQualityScore = 0;

    if (labelIndices_.isPartial()) {
        for (uint32 c = 0; c < numPredictions; c++) {
            sumOfQualityScores += 1 - qualityScoreIterator[c];
        }

        bestQualityScore = 1 - (sumOfQualityScores / numPredictions) * liftFunctionPtr_->calculateLift(numPredictions);
        bestNumPredictions = numPredictions;
    } else {
        sortedIndices = argsort(qualityScoreIterator, numPredictions);
        float64 maximumLift = liftFunctionPtr_->getMaxLift();

        for (uint32 c = 0; c < numPredictions; c++) {
            sumOfQualityScores += 1 - qualityScoreIterator[sortedIndices[c]];
            float64 qualityScore = 1 - (sumOfQualityScores / (c + 1)) * liftFunctionPtr_->calculateLift(c + 1);

            if (c == 0 || qualityScore < bestQualityScore) {
                bestNumPredictions = c + 1;
                bestQualityScore = qualityScore;
            }

            if (qualityScore * maximumLift < bestQualityScore) {
                // Prunable by decomposition...
                break;
            }
        }
    }

    if (bestHead == nullptr || bestQualityScore < bestHead->overallQualityScore) {
        if (headPtr_.get() == nullptr) {
            headPtr_ = std::make_unique<PartialPrediction>(bestNumPredictions);
        } else if (headPtr_->getNumElements() != bestNumPredictions) {
            headPtr_->setNumElements(bestNumPredictions);
        }

        typename T::index_const_iterator indexIterator = labelIndices_.indices_cbegin();
        PartialPrediction::iterator headValueIterator = headPtr_->begin();
        PartialPrediction::index_iterator headIndexIterator = headPtr_->indices_begin();

        if (labelIndices_.isPartial()) {
            for (uint32 c = 0; c < bestNumPredictions; c++) {
                headIndexIterator[c] = indexIterator[c];
                headValueIterator[c] = valueIterator[c];
            }
        } else {
            for (uint32 c = 0; c < bestNumPredictions; c++) {
                uint32 i = sortedIndices[c];
                headIndexIterator[c] = indexIterator[i];
                headValueIterator[c] = valueIterator[i];
            }
        }

        headPtr_->overallQualityScore = bestQualityScore;
        result = headPtr_.get();
    }

    delete[] sortedIndices;
    return result;
}

template<class T>
std::unique_ptr<AbstractEvaluatedPrediction> PartialHeadRefinementImpl<T>::pollHead() {
    return std::move(headPtr_);
}

template<class T>
const EvaluatedPrediction& PartialHeadRefinementImpl<T>::calculatePrediction(IStatisticsSubset& statisticsSubset,
                                                                             bool uncovered, bool accumulated) const {
    return statisticsSubset.calculateLabelWisePrediction(uncovered, accumulated);
}

PartialHeadRefinementFactoryImpl::PartialHeadRefinementFactoryImpl(std::shared_ptr<ILiftFunction> liftFunctionPtr)
    : liftFunctionPtr_(liftFunctionPtr) {

}

std::unique_ptr<IHeadRefinement> PartialHeadRefinementFactoryImpl::create(const RangeIndexVector& labelIndices) const {
    return std::make_unique<PartialHeadRefinementImpl<RangeIndexVector>>(labelIndices, liftFunctionPtr_);
}

std::unique_ptr<IHeadRefinement> PartialHeadRefinementFactoryImpl::create(const DenseIndexVector& labelIndices) const {
    return std::make_unique<PartialHeadRefinementImpl<DenseIndexVector>>(labelIndices, liftFunctionPtr_);
}
