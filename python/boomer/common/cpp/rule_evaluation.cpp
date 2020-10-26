#include "rule_evaluation.h"


EvaluatedPrediction::EvaluatedPrediction(uint32 numElements)
    : DenseVector<float64>(numElements) {

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
    return &qualityScores_[this->getNumElements()];
}

LabelWiseEvaluatedPrediction::quality_score_const_iterator LabelWiseEvaluatedPrediction::quality_scores_cbegin() const {
    return qualityScores_;
}

LabelWiseEvaluatedPrediction::quality_score_const_iterator LabelWiseEvaluatedPrediction::quality_scores_cend() const {
    return &qualityScores_[this->getNumElements()];
}
