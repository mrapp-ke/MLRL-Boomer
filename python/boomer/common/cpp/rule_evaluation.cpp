#include "rule_evaluation.h"


EvaluatedPrediction::EvaluatedPrediction(uint32 numElements)
    : scoreVector_(DenseVector<float64>(numElements)) {

}

uint32 EvaluatedPrediction::getNumElements() const {
    return scoreVector_.getNumElements();
}

EvaluatedPrediction::iterator EvaluatedPrediction::begin() {
    return scoreVector_.begin();
}

EvaluatedPrediction::iterator EvaluatedPrediction::end() {
    return scoreVector_.end();
}

EvaluatedPrediction::const_iterator EvaluatedPrediction::cbegin() const {
    return scoreVector_.cbegin();
}

EvaluatedPrediction::const_iterator EvaluatedPrediction::cend() const {
    return scoreVector_.cend();
}

LabelWiseEvaluatedPrediction::LabelWiseEvaluatedPrediction(uint32 numElements)
    : EvaluatedPrediction(numElements), qualityScoreVector_(DenseVector<float64>(numElements)) {

}

LabelWiseEvaluatedPrediction::quality_score_iterator LabelWiseEvaluatedPrediction::quality_scores_begin() {
    return qualityScoreVector_.begin();
}

LabelWiseEvaluatedPrediction::quality_score_iterator LabelWiseEvaluatedPrediction::quality_scores_end() {
    return qualityScoreVector_.end();
}

LabelWiseEvaluatedPrediction::quality_score_const_iterator LabelWiseEvaluatedPrediction::quality_scores_cbegin() const {
    return qualityScoreVector_.cbegin();
}

LabelWiseEvaluatedPrediction::quality_score_const_iterator LabelWiseEvaluatedPrediction::quality_scores_cend() const {
    return qualityScoreVector_.cend();
}
