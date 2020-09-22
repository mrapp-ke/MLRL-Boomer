#include "head_refinement.h"
#include "../../common/cpp/tuples.h"
#include <stdlib.h>

using namespace seco;


static inline uint32* argsort(float64* a, uint32 numElements) {
    IndexedFloat64* tmpArray = (IndexedFloat64*) malloc(numElements * sizeof(IndexedFloat64));

    for (uint32 i = 0; i < numElements; i++) {
        tmpArray[i].index = i;
        tmpArray[i].value = a[i];
    }

    qsort(tmpArray, numElements, sizeof(IndexedFloat64), &tuples::compareIndexedFloat64);
    uint32* sortedArray = (uint32*) malloc(numElements * sizeof(uint32));

    for (uint32 i = 0; i < numElements; i++) {
        sortedArray[i] = tmpArray[i].index;
    }

    free(tmpArray);
    return sortedArray;
}

PartialHeadRefinementImpl::PartialHeadRefinementImpl(std::shared_ptr<AbstractLiftFunction> liftFunctionPtr) {
    liftFunctionPtr_ = liftFunctionPtr;
}

PredictionCandidate* PartialHeadRefinementImpl::findHead(PredictionCandidate* bestHead,
                                                         PredictionCandidate* recyclableHead,
                                                         const uint32* labelIndices,
                                                         AbstractStatisticsSubset* statisticsSubset, bool uncovered,
                                                         bool accumulated) {
    PredictionCandidate* result = NULL;
    AbstractLiftFunction* liftFunction = liftFunctionPtr_.get();
    LabelWisePredictionCandidate* prediction = statisticsSubset->calculateLabelWisePrediction(uncovered, accumulated);
    uint32 numPredictions = prediction->numPredictions_;
    float64* predictedScores = prediction->predictedScores_;
    float64* qualityScores = prediction->qualityScores_;
    uint32* sortedIndices = NULL;
    float64 sumOfQualityScores = 0;
    uint32 bestNumPredictions = 0;
    float64 bestQualityScore = 0;

    if (labelIndices == NULL) {
        sortedIndices = argsort(qualityScores, numPredictions);
        float64 maximumLift = liftFunction->getMaxLift();

        for (uint32 c = 0; c < numPredictions; c++) {
            sumOfQualityScores += 1 - qualityScores[sortedIndices[c]];
            float64 qualityScore = 1 - (sumOfQualityScores / (c + 1)) * liftFunction->calculateLift(c + 1);

            if (c == 0 || qualityScore < bestQualityScore) {
                bestNumPredictions = c + 1;
                bestQualityScore = qualityScore;
            }

            if (qualityScore * maximumLift < bestQualityScore) {
                // Prunable by decomposition...
                break;
            }
        }
    } else {
        for (uint32 c = 0; c < numPredictions; c++) {
            sumOfQualityScores += 1 - qualityScores[c];
        }

        bestQualityScore = 1 - (sumOfQualityScores / numPredictions) * liftFunction->calculateLift(numPredictions);
        bestNumPredictions = numPredictions;
    }

    if (bestHead == NULL || bestQualityScore < bestHead->overallQualityScore_) {
        if (recyclableHead == NULL) {
            // Create a new `PredictionCandidate` and return it...
            uint32* candidateLabelIndices = (uint32*) malloc(bestNumPredictions * sizeof(uint32));
            float64* candidatePredictedScores = (float64*) malloc(bestNumPredictions * sizeof(float64));

            if (labelIndices == NULL) {
                for (uint32 c = 0; c < bestNumPredictions; c++) {
                    uint32 i = sortedIndices[c];
                    candidateLabelIndices[c] = labelIndices == NULL ? i : labelIndices[i];
                    candidatePredictedScores[c] = predictedScores[i];
                }
            } else {
                for (uint32 c = 0; c < bestNumPredictions; c++) {
                    candidateLabelIndices[c] = labelIndices[c];
                    candidatePredictedScores[c] = predictedScores[c];
                }
            }

            result = new PredictionCandidate(bestNumPredictions, candidateLabelIndices, candidatePredictedScores,
                                             bestQualityScore);
        } else {
            // Modify the `recyclableHead` and return it...
            if (recyclableHead->numPredictions_ != bestNumPredictions) {
                recyclableHead->numPredictions_ = bestNumPredictions;
                recyclableHead->labelIndices_ = (uint32*) realloc(recyclableHead->labelIndices_,
                                                                  bestNumPredictions * sizeof(uint32));
                recyclableHead->predictedScores_ = (float64*) realloc(recyclableHead->predictedScores_,
                                                                      bestNumPredictions * sizeof(float64));
            }

            if (labelIndices == NULL) {
                for (uint32 c = 0; c < bestNumPredictions; c++) {
                    uint32 i = sortedIndices[c];
                    recyclableHead->labelIndices_[c] = labelIndices == NULL ? i : labelIndices[i];
                    recyclableHead->predictedScores_[c] = predictedScores[i];
                }
            } else {
                for (uint32 c = 0; c < bestNumPredictions; c++) {
                    recyclableHead->labelIndices_[c] = labelIndices[c];
                    recyclableHead->predictedScores_[c] = predictedScores[c];
                }
            }

            recyclableHead->overallQualityScore_ = bestQualityScore;
            result = recyclableHead;
        }
    }

    free(sortedIndices);
    return result;
}

PredictionCandidate* PartialHeadRefinementImpl::calculatePrediction(AbstractStatisticsSubset* statisticsSubset,
                                                                    bool uncovered, bool accumulated) {
    return statisticsSubset->calculateLabelWisePrediction(uncovered, accumulated);
}
