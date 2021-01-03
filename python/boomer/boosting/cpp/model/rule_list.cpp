#include "rule_list.h"
#include "../../../common/cpp/model/rule.h"
#include "../../../common/cpp/model/body_empty.h"
#include "../../../common/cpp/model/body_conjunctive.h"
#include <list>


/**
 * A model that stores several rules in a list.
 */
class RuleList final : public IModel {

    private:

        std::list<std::unique_ptr<Rule>> list_;

    public:

        /**
         * Adds a new rule to the model.
         *
         * @param rulePtr An unique pointer to an object of type `Rule` that should be added
         */
        void append(std::unique_ptr<Rule> rulePtr) {
            list_.push_back(std::move(rulePtr));
        }

        void predict(const CContiguousFeatureMatrix& featureMatrix,
                     DenseMatrix<float64>& predictionMatrix) const override {
            for (auto it = list_.cbegin(); it != list_.cend(); it++) {
                const Rule& rule = **it;
                rule.predict(featureMatrix, predictionMatrix);
            }
        }

        void predict(const CsrFeatureMatrix& featureMatrix, DenseMatrix<float64>& predictionMatrix) const override {
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

};

void RuleListBuilder::setDefaultRule(const AbstractPrediction* prediction) {
    if (prediction != nullptr) {
        modelPtr_->append(std::make_unique<Rule>(std::make_unique<EmptyBody>(), prediction->toHead()));
    }
}

void RuleListBuilder::addRule(const ConditionList& conditions, const AbstractPrediction& prediction) {
    modelPtr_->append(std::make_unique<Rule>(std::make_unique<ConjunctiveBody>(conditions), prediction.toHead()));
}

std::unique_ptr<IModel> RuleListBuilder::build() {
    return std::move(modelPtr_);
}
