#include "rule_evaluation.h"


AbstractDefaultRuleEvaluation::~AbstractDefaultRuleEvaluation() {

}

Prediction* AbstractDefaultRuleEvaluation::calculateDefaultPrediction(AbstractLabelMatrix* labelMatrix) {
    return NULL;
}
