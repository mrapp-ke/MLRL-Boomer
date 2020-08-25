#include "rule_evaluation.h"


AbstractDefaultRuleEvaluation::~AbstractDefaultRuleEvaluation() {

}

Prediction* AbstractDefaultRuleEvaluation::calculateDefaultPrediction(
        AbstractRandomAccessLabelMatrix* labelMatrix) {
    return NULL;
}
