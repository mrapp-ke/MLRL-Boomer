#include "head_refinement.h"
#include <stdlib.h>


bool SingleLabelHeadRefinementImpl::findHead(PredictionCandidate* bestHead,
                                             std::unique_ptr<PredictionCandidate>& headPtr, const uint32* labelIndices,
                                             IStatisticsSubset& statisticsSubset, bool uncovered, bool accumulated) {
    LabelWisePredictionCandidate& prediction = statisticsSubset.calculateLabelWisePrediction(uncovered, accumulated);
    uint32 numPredictions = prediction.numPredictions_;
    float64* qualityScores = prediction.qualityScores_;
    uint32 bestC = 0;
    float64 bestQualityScore = qualityScores[bestC];

    for (uint32 c = 1; c < numPredictions; c++) {
        float64 qualityScore = qualityScores[c];

        if (qualityScore < bestQualityScore) {
            bestQualityScore = qualityScore;
            bestC = c;
        }
    }

    // The quality score must be better than that of `bestHead`...
    if (bestHead == NULL || bestQualityScore < bestHead->overallQualityScore_) {
        float64* predictedScores = prediction.predictedScores_;
        PredictionCandidate* recyclableHead = headPtr.get();

        if (recyclableHead == NULL) {
            // Create a new `PredictionCandidate`...
            uint32* candidateLabelIndices = (uint32*) malloc(sizeof(uint32));
            candidateLabelIndices[0] = labelIndices == NULL ? bestC : labelIndices[bestC];
            float64* candidatePredictedScores = (float64*) malloc(sizeof(float64));
            candidatePredictedScores[0] = predictedScores[bestC];
            headPtr = std::make_unique<PredictionCandidate>(1, candidateLabelIndices, candidatePredictedScores,
                                                            bestQualityScore);
        } else {
            // Modify the `recyclableHead`...
            recyclableHead->labelIndices_[0] = labelIndices == NULL ? bestC : labelIndices[bestC];
            recyclableHead->predictedScores_[0] = predictedScores[bestC];
            recyclableHead->overallQualityScore_ = bestQualityScore;
        }

        return true;
    }

    return false;
}

PredictionCandidate& SingleLabelHeadRefinementImpl::calculatePrediction(IStatisticsSubset& statisticsSubset,
                                                                        bool uncovered, bool accumulated) {
    return statisticsSubset.calculateLabelWisePrediction(uncovered, accumulated);
}

bool FullHeadRefinementImpl::findHead(PredictionCandidate* bestHead, std::unique_ptr<PredictionCandidate>& headPtr,
                                      const uint32* labelIndices, IStatisticsSubset& statisticsSubset, bool uncovered,
                                      bool accumulated) {
    PredictionCandidate& prediction = statisticsSubset.calculateExampleWisePrediction(uncovered, accumulated);
    float64 overallQualityScore = prediction.overallQualityScore_;

    // The quality score must be better than that of `bestHead`...
    if (bestHead == NULL || overallQualityScore < bestHead->overallQualityScore_) {
        uint32 numPredictions = prediction.numPredictions_;
        float64* predictedScores = prediction.predictedScores_;
        PredictionCandidate* recyclableHead = headPtr.get();

        if (recyclableHead == NULL) {
            // Create a new `PredictionCandidate`...
            float64* candidatePredictedScores = (float64*) malloc(numPredictions * sizeof(float64));
            uint32* candidateLabelIndices = NULL;

            for (uint32 c = 0; c < numPredictions; c++) {
                candidatePredictedScores[c] = predictedScores[c];
            }

            if (labelIndices != NULL) {
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
                recyclableHead->predictedScores_[c] = predictedScores[c];
            }

            recyclableHead->overallQualityScore_ = overallQualityScore;
        }

        return true;
    }

    return false;
}

PredictionCandidate& FullHeadRefinementImpl::calculatePrediction(IStatisticsSubset& statisticsSubset, bool uncovered,
                                                                 bool accumulated) {
    return statisticsSubset.calculateExampleWisePrediction(uncovered, accumulated);
}
