#include "rule_evaluation.h"


EvaluatedPrediction::EvaluatedPrediction(uint32 numElements)
    : predictedScoreVector_(DenseVector<float64>(numElements)) {

}

EvaluatedPrediction::iterator EvaluatedPrediction::begin() {
    return predictedScoreVector_.begin();
}

EvaluatedPrediction::iterator EvaluatedPrediction::end() {
    return predictedScoreVector_.end();
}

EvaluatedPrediction::const_iterator EvaluatedPrediction::cbegin() const {
    return predictedScoreVector_.cbegin();
}

EvaluatedPrediction::const_iterator EvaluatedPrediction::cend() const {
    return predictedScoreVector_.cend();
}

uint32 EvaluatedPrediction::getNumElements() const {
    return predictedScoreVector_.getNumElements();
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
