#include "rule_evaluation.h"
#include <stdlib.h>


DefaultPrediction::DefaultPrediction(intp numPredictions, float64* predictedScores) {
    numPredictions_ = numPredictions;
    predictedScores_ = predictedScores;
}

DefaultPrediction::~DefaultPrediction() {
    free(predictedScores_);
}

Prediction::Prediction(intp numPredictions, float64* predictedScores, float64 overallQualityScore)
    : DefaultPrediction(numPredictions, predictedScores) {
    overallQualityScore_ = overallQualityScore;
}

LabelWisePrediction::LabelWisePrediction(intp numPredictions, float64* predictedScores, float64* qualityScores,
                                         float64 overallQualityScore)
    : Prediction(numPredictions, predictedScores, overallQualityScore) {
    qualityScores_ = qualityScores;
}

LabelWisePrediction::~LabelWisePrediction() {
    free(qualityScores_);
}

AbstractDefaultRuleEvaluation::~AbstractDefaultRuleEvaluation() {

}

DefaultPrediction* AbstractDefaultRuleEvaluation::calculateDefaultPrediction(AbstractLabelMatrix* labelMatrix) {
    return NULL;
}
