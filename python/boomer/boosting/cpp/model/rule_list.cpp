#include "rule_list.h"


void RuleList::append(std::unique_ptr<Rule> rulePtr) {
    list_.push_back(std::move(rulePtr));
}

void RuleList::predict(const CContiguousFeatureMatrix& featureMatrix, DenseMatrix<float64>& predictionMatrix) const {
    for (auto it = list_.cbegin(); it != list_.cend(); it++) {
        const Rule& rule = **it;
        rule.predict(featureMatrix, predictionMatrix);
    }
}

void RuleList::predict(const CsrFeatureMatrix& featureMatrix, DenseMatrix<float64>& predictionMatrix) const {
    uint32 numFeatures = featureMatrix.getNumFeatures();
    float32 tmpArray1[numFeatures];
    uint32 tmpArray2[numFeatures] = {};
    uint32 n = 1;

    for (auto it = list_.cbegin(); it != list_.cend(); it++) {
        const Rule& rule = **it;
        rule.predict(featureMatrix, predictionMatrix, &tmpArray1[0], &tmpArray2[0], n);
        n++;
    }
}
