#include "head_refinement.h"
#include <cstdlib>


template<class T>
SingleLabelHeadRefinementImpl<T>::SingleLabelHeadRefinementImpl(const T& labelIndices)
    : labelIndices_(labelIndices) {

}

template<class T>
const PredictionCandidate* SingleLabelHeadRefinementImpl<T>::findHead(const PredictionCandidate* bestHead,
                                                                      const uint32* labelIndices,
                                                                      IStatisticsSubset& statisticsSubset,
                                                                      bool uncovered, bool accumulated) {
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

        if (headPtr_.get() == nullptr) {
            headPtr_ = std::make_unique<PartialPrediction>(1);

            // TODO Remove the following
            uint32* candidateLabelIndices = (uint32*) malloc(sizeof(uint32));
            float64* candidatePredictedScores = (float64*) malloc(sizeof(float64));
            headPtr_->labelIndices_ = candidateLabelIndices;
            headPtr_->predictedScores_ = candidatePredictedScores;
        }

        // TODO Remove the following
        headPtr_->predictedScores_[0] = valueIterator[bestC];
        headPtr_->labelIndices_[0] = labelIndices == nullptr ? bestC : labelIndices[bestC];

        PartialPrediction::iterator headValueIterator = headPtr_->begin();
        PartialPrediction::index_iterator headIndexIterator = headPtr_->indices_begin();
        headValueIterator[0] = valueIterator[bestC];
        headIndexIterator[0] = labelIndices == nullptr ? bestC : labelIndices[bestC];
        headPtr_->overallQualityScore = bestQualityScore;
        return headPtr_.get();
    }

    return nullptr;
}

template<class T>
std::unique_ptr<PredictionCandidate> SingleLabelHeadRefinementImpl<T>::pollHead() {
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
const PredictionCandidate* FullHeadRefinementImpl<T>::findHead(const PredictionCandidate* bestHead,
                                                               const uint32* labelIndices,
                                                               IStatisticsSubset& statisticsSubset, bool uncovered,
                                                               bool accumulated) {
    const EvaluatedPrediction& prediction = statisticsSubset.calculateExampleWisePrediction(uncovered, accumulated);
    float64 overallQualityScore = prediction.overallQualityScore;

    // The quality score must be better than that of `bestHead`...
    if (bestHead == nullptr || overallQualityScore < bestHead->overallQualityScore) {
        uint32 numPredictions = prediction.getNumElements();
        EvaluatedPrediction::const_iterator valueIterator = prediction.cbegin();

        if (headPtr_.get() == nullptr) {
            if (labelIndices != nullptr) {
                std::unique_ptr<PartialPrediction> headPtr = std::make_unique<PartialPrediction>(numPredictions);
                PartialPrediction::index_iterator headIndexIterator = headPtr->indices_begin();

                for (uint32 c = 0; c < numPredictions; c++) {
                    headIndexIterator[c] = labelIndices[c];
                }

                // TODO Remove the following
                uint32* candidateLabelIndices = (uint32*) malloc(numPredictions * sizeof(uint32));
                headPtr_->labelIndices_ = candidateLabelIndices;

                for (uint32 c = 0; c < numPredictions; c++) {
                    candidateLabelIndices[c] = labelIndices[c];
                }

                headPtr_ = std::move(headPtr);
            } else {
                headPtr_ = std::make_unique<FullPrediction>(numPredictions);
            }

            // TODO Remove the following
            float64* candidatePredictedScores = (float64*) malloc(numPredictions * sizeof(float64));
            headPtr_->predictedScores_ = candidatePredictedScores;
        }

        // TODO Remove the following
        for (uint32 c = 0; c < numPredictions; c++) {
            headPtr_->predictedScores_[c] = valueIterator[c];
        }

        PredictionCandidate::iterator headValueIterator = headPtr_->begin();

        for (uint32 c = 0; c < numPredictions; c++) {
            headValueIterator[c] = valueIterator[c];
        }

        headPtr_->overallQualityScore = overallQualityScore;
        return headPtr_.get();
    }

    return nullptr;
}

template<class T>
std::unique_ptr<PredictionCandidate> FullHeadRefinementImpl<T>::pollHead() {
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
