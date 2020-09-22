#include "head_refinement.h"
#include <stdlib.h>


PredictionCandidate* SingleLabelHeadRefinementImpl::findHead(PredictionCandidate* bestHead,
                                                             PredictionCandidate* recyclableHead,
                                                             const uint32* labelIndices,
                                                             AbstractStatisticsSubset* statisticsSubset, bool uncovered,
                                                             bool accumulated) {
    LabelWisePredictionCandidate* prediction = statisticsSubset->calculateLabelWisePrediction(uncovered, accumulated);
    uint32 numPredictions = prediction->numPredictions_;
    float64* qualityScores = prediction->qualityScores_;
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
        float64* predictedScores = prediction->predictedScores_;

        if (recyclableHead == NULL) {
            // Create a new `PredictionCandidate` and return it...
            uint32* candidateLabelIndices = (uint32*) malloc(sizeof(uint32));
            candidateLabelIndices[0] = labelIndices == NULL ? bestC : labelIndices[bestC];
            float64* candidatePredictedScores = (float64*) malloc(sizeof(float64));
            candidatePredictedScores[0] = predictedScores[bestC];
            return new PredictionCandidate(1, candidateLabelIndices, candidatePredictedScores, bestQualityScore);
        } else {
            // Modify the `recyclableHead` and return it...
            recyclableHead->labelIndices_[0] = labelIndices == NULL ? bestC : labelIndices[bestC];
            recyclableHead->predictedScores_[0] = predictedScores[bestC];
            recyclableHead->overallQualityScore_ = bestQualityScore;
            return recyclableHead;
        }
    }

    // Return NULL, as the quality score of the head that has been found is worse than that of `bestHead`...
    return NULL;
}

PredictionCandidate* SingleLabelHeadRefinementImpl::calculatePrediction(AbstractStatisticsSubset* statisticsSubset,
                                                                        bool uncovered, bool accumulated) {
    return statisticsSubset->calculateLabelWisePrediction(uncovered, accumulated);
}

PredictionCandidate* FullHeadRefinementImpl::findHead(PredictionCandidate* bestHead,
                                                      PredictionCandidate* recyclableHead, const uint32* labelIndices,
                                                      AbstractStatisticsSubset* statisticsSubset, bool uncovered,
                                                      bool accumulated) {
    PredictionCandidate* prediction = statisticsSubset->calculateExampleWisePrediction(uncovered, accumulated);
    float64 overallQualityScore = prediction->overallQualityScore_;

    // The quality score must be better than that of `bestHead`...
    if (bestHead == NULL || overallQualityScore < bestHead->overallQualityScore_) {
        uint32 numPredictions = prediction->numPredictions_;
        float64* predictedScores = prediction->predictedScores_;

        if (recyclableHead == NULL) {
            // Create a new `PredictionCandidate` and return it...
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

            return new PredictionCandidate(numPredictions, candidateLabelIndices, candidatePredictedScores,
                                           overallQualityScore);
        } else {
            // Modify the `recyclableHead` and return it...
            for (uint32 c = 0; c < numPredictions; c++) {
                recyclableHead->predictedScores_[c] = predictedScores[c];
            }

            recyclableHead->overallQualityScore_ = overallQualityScore;
            return recyclableHead;
        }
    }

    // Return NULL, as the quality score of the head that has been found is worse than that of `bestHead`...
    return NULL;
}

PredictionCandidate* FullHeadRefinementImpl::calculatePrediction(AbstractStatisticsSubset* statisticsSubset,
                                                                 bool uncovered, bool accumulated) {
    return statisticsSubset->calculateExampleWisePrediction(uncovered, accumulated);
}
