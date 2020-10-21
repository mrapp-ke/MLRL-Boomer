#include "rule_evaluation.h"


EvaluatedPrediction::EvaluatedPrediction(uint32 numElements)
    : numElements_(numElements), scores_(new float64[numElements]) {

}

EvaluatedPrediction::~EvaluatedPrediction() {
    delete[] scores_;
}

uint32 EvaluatedPrediction::getNumElements() const {
    return numElements_;
}

EvaluatedPrediction::iterator EvaluatedPrediction::begin() {
    return &scores_[0];
}

EvaluatedPrediction::iterator EvaluatedPrediction::end() {
    return &scores_[numElements_];
}

EvaluatedPrediction::const_iterator EvaluatedPrediction::cbegin() const {
    return &scores_[0];
}

EvaluatedPrediction::const_iterator EvaluatedPrediction::cend() const {
    return &scores_[numElements_];
}

LabelWiseEvaluatedPrediction::LabelWiseEvaluatedPrediction(uint32 numElements)
    : EvaluatedPrediction(numElements), qualityScores_(new float64[numElements]) {

}

LabelWiseEvaluatedPrediction::~LabelWiseEvaluatedPrediction() {
    delete[] qualityScores_;
}

LabelWiseEvaluatedPrediction::quality_score_iterator LabelWiseEvaluatedPrediction::quality_scores_begin() {
    return qualityScores_;
}

LabelWiseEvaluatedPrediction::quality_score_iterator LabelWiseEvaluatedPrediction::quality_scores_end() {
    return &qualityScores_[numElements_];
}

LabelWiseEvaluatedPrediction::quality_score_const_iterator LabelWiseEvaluatedPrediction::quality_scores_cbegin() const {
    return qualityScores_;
}

LabelWiseEvaluatedPrediction::quality_score_const_iterator LabelWiseEvaluatedPrediction::quality_scores_cend() const {
    return &qualityScores_[numElements_];
}
