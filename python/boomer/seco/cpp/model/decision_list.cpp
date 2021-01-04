#include "decision_list.h"
#include "../../../common/cpp/model/rule.h"
#include "../../../common/cpp/model/body_empty.h"
#include "../../../common/cpp/model/body_conjunctive.h"
#include <list>


/**
 * A model that stores several rules in a decision list.
 */
class DecisionList final : public IModel {

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
                     CContiguousView<float64>& predictionMatrix) const override {
            uint32 numExamples = predictionMatrix.getNumRows();
            uint32 numLabels = predictionMatrix.getNumCols();
            PredictionMask mask(numExamples, numLabels, true);

            for (auto it = list_.cbegin(); it != list_.cend(); it++) {
                const Rule& rule = **it;
                rule.predict(featureMatrix, predictionMatrix, mask);
            }
        }

        void predict(const CsrFeatureMatrix& featureMatrix, CContiguousView<float64>& predictionMatrix) const override {
            uint32 numFeatures = featureMatrix.getNumCols();
            uint32 numExamples = predictionMatrix.getNumRows();
            uint32 numLabels = predictionMatrix.getNumCols();
            PredictionMask mask(numExamples, numLabels, true);
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

void DecisionListBuilder::setDefaultRule(const AbstractPrediction& prediction) {
    defaultRulePtr_ = std::make_unique<Rule>(std::make_unique<EmptyBody>(), prediction.toHead());
}

void DecisionListBuilder::addRule(const ConditionList& conditions, const AbstractPrediction& prediction) {
    modelPtr_->append(std::make_unique<Rule>(std::make_unique<ConjunctiveBody>(conditions), prediction.toHead()));
}

std::unique_ptr<IModel> DecisionListBuilder::build() {
    if (defaultRulePtr_.get() != nullptr) {
        modelPtr_->append(std::move(defaultRulePtr_));
    }

    return std::move(modelPtr_);
}
