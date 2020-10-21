#include "head_refinement.h"
#include <cstdlib>


bool SingleLabelHeadRefinementImpl::findHead(const PredictionCandidate* bestHead,
                                             std::unique_ptr<PredictionCandidate>& headPtr, const uint32* labelIndices,
                                             IStatisticsSubset& statisticsSubset, bool uncovered,
                                             bool accumulated) const {
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
        PredictionCandidate* recyclableHead = headPtr.get();

        if (recyclableHead == nullptr) {
            // Create a new `PredictionCandidate`...
            uint32* candidateLabelIndices = (uint32*) malloc(sizeof(uint32));
            candidateLabelIndices[0] = labelIndices == nullptr ? bestC : labelIndices[bestC];
            float64* candidatePredictedScores = (float64*) malloc(sizeof(float64));
            candidatePredictedScores[0] = valueIterator[bestC];
            headPtr = std::make_unique<PredictionCandidate>(1, candidateLabelIndices, candidatePredictedScores,
                                                            bestQualityScore);
        } else {
            // Modify the `recyclableHead`...
            recyclableHead->labelIndices_[0] = labelIndices == nullptr ? bestC : labelIndices[bestC];
            recyclableHead->predictedScores_[0] = valueIterator[bestC];
            recyclableHead->overallQualityScore_ = bestQualityScore;
        }

        return true;
    }

    return false;
}

const EvaluatedPrediction& SingleLabelHeadRefinementImpl::calculatePrediction(IStatisticsSubset& statisticsSubset,
                                                                              bool uncovered, bool accumulated) const {
    return statisticsSubset.calculateLabelWisePrediction(uncovered, accumulated);
}

std::unique_ptr<IHeadRefinement> SingleLabelHeadRefinementFactoryImpl::create() const {
    return std::make_unique<SingleLabelHeadRefinementImpl>();
}

bool FullHeadRefinementImpl::findHead(const PredictionCandidate* bestHead, std::unique_ptr<PredictionCandidate>& headPtr,
                                      const uint32* labelIndices, IStatisticsSubset& statisticsSubset, bool uncovered,
                                      bool accumulated) const {
    const EvaluatedPrediction& prediction = statisticsSubset.calculateExampleWisePrediction(uncovered, accumulated);
    float64 overallQualityScore = prediction.overallQualityScore;

    // The quality score must be better than that of `bestHead`...
    if (bestHead == nullptr || overallQualityScore < bestHead->overallQualityScore_) {
        uint32 numPredictions = prediction.getNumElements();
        EvaluatedPrediction::const_iterator valueIterator = prediction.cbegin();
        PredictionCandidate* recyclableHead = headPtr.get();

        if (recyclableHead == nullptr) {
            // Create a new `PredictionCandidate`...
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

            headPtr = std::make_unique<PredictionCandidate>(numPredictions, candidateLabelIndices,
                                                            candidatePredictedScores, overallQualityScore);
        } else {
            // Modify the `recyclableHead`...
            for (uint32 c = 0; c < numPredictions; c++) {
                recyclableHead->predictedScores_[c] = valueIterator[c];
            }

            recyclableHead->overallQualityScore_ = overallQualityScore;
        }

        return true;
    }

    return false;
}

const EvaluatedPrediction& FullHeadRefinementImpl::calculatePrediction(IStatisticsSubset& statisticsSubset,
                                                                       bool uncovered, bool accumulated) const {
    return statisticsSubset.calculateExampleWisePrediction(uncovered, accumulated);
}

std::unique_ptr<IHeadRefinement> FullHeadRefinementFactoryImpl::create() const {
    return std::make_unique<FullHeadRefinementImpl>();
}