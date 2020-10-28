#include "head_refinement.h"
#include <cstdlib>


template<class T>
SingleLabelHeadRefinementImpl<T>::SingleLabelHeadRefinementImpl(const T& labelIndices)
    : labelIndices_(labelIndices) {

}

template<class T>
const AbstractEvaluatedPrediction* SingleLabelHeadRefinementImpl<T>::findHead(
        const AbstractEvaluatedPrediction* bestHead, IStatisticsSubset& statisticsSubset, bool uncovered,
        bool accumulated) {
    const LabelWiseEvaluatedPrediction& prediction = statisticsSubset.calculateLabelWisePrediction(uncovered,
                                                                                                   accumulated);
    uint32 numPredictions = prediction.getNumElements();
    LabelWiseEvaluatedPrediction::quality_score_const_iterator qualityScoreIterator =
        prediction.quality_scores_cbegin();
    uint32 bestC = 0;
    float64 bestQualityScore = qualityScoreIterator[bestC];

    for (uint32 c = 1; c < numPredictions; c++) {
        float64 qualityScore = qualityScoreIterator[c];

        if (qualityScore < bestQualityScore) {
            bestQualityScore = qualityScore;
            bestC = c;
        }
    }

    // The quality score must be better than that of `bestHead`...
    if (bestHead == nullptr || bestQualityScore < bestHead->overallQualityScore) {
        LabelWiseEvaluatedPrediction::const_iterator valueIterator = prediction.cbegin();
        typename T::index_const_iterator indexIterator = labelIndices_.indices_cbegin();

        if (headPtr_.get() == nullptr) {
            headPtr_ = std::make_unique<PartialPrediction>(1);
        }

        PartialPrediction::iterator headValueIterator = headPtr_->begin();
        PartialPrediction::index_iterator headIndexIterator = headPtr_->indices_begin();
        headValueIterator[0] = valueIterator[bestC];
        headIndexIterator[0] = indexIterator[bestC];
        headPtr_->overallQualityScore = bestQualityScore;
        return headPtr_.get();
    }

    return nullptr;
}

template<class T>
std::unique_ptr<AbstractEvaluatedPrediction> SingleLabelHeadRefinementImpl<T>::pollHead() {
    return std::move(headPtr_);
}

template<class T>
const EvaluatedPrediction& SingleLabelHeadRefinementImpl<T>::calculatePrediction(IStatisticsSubset& statisticsSubset,
                                                                                 bool uncovered,
                                                                                 bool accumulated) const {
    return statisticsSubset.calculateLabelWisePrediction(uncovered, accumulated);
}

std::unique_ptr<IHeadRefinement> SingleLabelHeadRefinementFactoryImpl::create(
        const RangeIndexVector& labelIndices) const {
    return std::make_unique<SingleLabelHeadRefinementImpl<RangeIndexVector>>(labelIndices);
}

std::unique_ptr<IHeadRefinement> SingleLabelHeadRefinementFactoryImpl::create(
        const DenseIndexVector& labelIndices) const {
    return std::make_unique<SingleLabelHeadRefinementImpl<DenseIndexVector>>(labelIndices);
}

template<class T>
FullHeadRefinementImpl<T>::FullHeadRefinementImpl(const T& labelIndices)
    : labelIndices_(labelIndices) {

}

template<class T>
const AbstractEvaluatedPrediction* FullHeadRefinementImpl<T>::findHead(const AbstractEvaluatedPrediction* bestHead,
                                                                       IStatisticsSubset& statisticsSubset,
                                                                       bool uncovered, bool accumulated) {
    const EvaluatedPrediction& prediction = statisticsSubset.calculateExampleWisePrediction(uncovered, accumulated);
    float64 overallQualityScore = prediction.overallQualityScore;

    // The quality score must be better than that of `bestHead`...
    if (bestHead == nullptr || overallQualityScore < bestHead->overallQualityScore) {
        uint32 numPredictions = prediction.getNumElements();
        EvaluatedPrediction::const_iterator valueIterator = prediction.cbegin();

        if (headPtr_.get() == nullptr) {
            if (labelIndices_.isPartial()) {
                typename T::index_const_iterator indexIterator = labelIndices_.indices_cbegin();
                std::unique_ptr<PartialPrediction> headPtr = std::make_unique<PartialPrediction>(numPredictions);
                PartialPrediction::index_iterator headIndexIterator = headPtr->indices_begin();

                for (uint32 c = 0; c < numPredictions; c++) {
                    headIndexIterator[c] = indexIterator[c];
                }

                headPtr_ = std::move(headPtr);
            } else {
                headPtr_ = std::make_unique<FullPrediction>(numPredictions);
            }
        }

        AbstractEvaluatedPrediction::iterator headValueIterator = headPtr_->begin();

        for (uint32 c = 0; c < numPredictions; c++) {
            headValueIterator[c] = valueIterator[c];
        }

        headPtr_->overallQualityScore = overallQualityScore;
        return headPtr_.get();
    }

    return nullptr;
}

template<class T>
std::unique_ptr<AbstractEvaluatedPrediction> FullHeadRefinementImpl<T>::pollHead() {
    return std::move(headPtr_);
}

template<class T>
const EvaluatedPrediction& FullHeadRefinementImpl<T>::calculatePrediction(IStatisticsSubset& statisticsSubset,
                                                                          bool uncovered, bool accumulated) const {
    return statisticsSubset.calculateExampleWisePrediction(uncovered, accumulated);
}

std::unique_ptr<IHeadRefinement> FullHeadRefinementFactoryImpl::create(const RangeIndexVector& labelIndices) const {
    return std::make_unique<FullHeadRefinementImpl<RangeIndexVector>>(labelIndices);
}

std::unique_ptr<IHeadRefinement> FullHeadRefinementFactoryImpl::create(const DenseIndexVector& labelIndices) const {
    return std::make_unique<FullHeadRefinementImpl<DenseIndexVector>>(labelIndices);
}
