#include "head_refinement.h"
#include <cstdlib>


template<class T>
SingleLabelHeadRefinementImpl<T>::SingleLabelHeadRefinementImpl(const T& labelIndices)
    : labelIndices_(labelIndices) {

}

template<class T>
bool SingleLabelHeadRefinementImpl<T>::findHead(const PredictionCandidate* bestHead,
                                                std::unique_ptr<PredictionCandidate>& headPtr,
                                                const uint32* labelIndices, IStatisticsSubset& statisticsSubset,
                                                bool uncovered, bool accumulated) const {
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
    if (bestHead == nullptr || bestQualityScore < bestHead->overallQualityScore_) {
        LabelWiseEvaluatedPrediction::const_iterator valueIterator = prediction.cbegin();

        if (headPtr.get() == nullptr) {
            headPtr = std::make_unique<PredictionCandidate>(1);

            // TODO Remove the following
            uint32* candidateLabelIndices = (uint32*) malloc(sizeof(uint32));
            float64* candidatePredictedScores = (float64*) malloc(sizeof(float64));
            headPtr->labelIndices_ = candidateLabelIndices;
            headPtr->predictedScores_ = candidatePredictedScores;
        }

        float64* headValueIterator = headPtr->predictedScores_;
        uint32* headIndexIterator = headPtr->labelIndices_;
        headValueIterator[0] = valueIterator[bestC];
        headIndexIterator[0] = labelIndices == nullptr ? bestC : labelIndices[bestC];
        headPtr->overallQualityScore_ = bestQualityScore;
        return true;
    }

    return false;
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
bool FullHeadRefinementImpl<T>::findHead(const PredictionCandidate* bestHead,
                                         std::unique_ptr<PredictionCandidate>& headPtr, const uint32* labelIndices,
                                         IStatisticsSubset& statisticsSubset, bool uncovered, bool accumulated) const {
    const EvaluatedPrediction& prediction = statisticsSubset.calculateExampleWisePrediction(uncovered, accumulated);
    float64 overallQualityScore = prediction.overallQualityScore;

    // The quality score must be better than that of `bestHead`...
    if (bestHead == nullptr || overallQualityScore < bestHead->overallQualityScore_) {
        uint32 numPredictions = prediction.getNumElements();
        EvaluatedPrediction::const_iterator valueIterator = prediction.cbegin();

        if (headPtr.get() == nullptr) {
            headPtr = std::make_unique<PredictionCandidate>(numPredictions);

            // TODO Remove the following
            float64* candidatePredictedScores = (float64*) malloc(numPredictions * sizeof(float64));
            uint32* candidateLabelIndices = nullptr;

            for (uint32 c = 0; c < numPredictions; c++) {
                candidatePredictedScores[c] = valueIterator[c];
            }

            if (labelIndices != nullptr) {
                uint32* candidateLabelIndices = (uint32*) malloc(numPredictions * sizeof(uint32));

                for (uint32 c = 0; c < numPredictions; c++) {
                    candidateLabelIndices[c] = labelIndices[c];
                }
            }

            headPtr->labelIndices_ = candidateLabelIndices;
            headPtr->predictedScores_ = candidatePredictedScores;
        }

        float64* headValueIterator = headPtr->predictedScores_;

        for (uint32 c = 0; c < numPredictions; c++) {
            headValueIterator[c] = valueIterator[c];
        }

        headPtr->overallQualityScore_ = overallQualityScore;
        return true;
    }

    return false;
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
