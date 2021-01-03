/**
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "../../../common/cpp/model/model_builder.h"

// Forward declarations
class DecisionList;
class Rule;

/**
 * Allows to build models that store several rules in the order they have been added, except for the default rule, which
 * is always located at the end. For prediction, the rules are processed in this particular order. Subsequent rules are
 * only allowed to predict for labels for which no previous rules has already provided a prediction.
 */
class DecisionListBuilder final : public IModelBuilder {

    private:

        std::unique_ptr<Rule> defaultRulePtr_;

        std::unique_ptr<DecisionList> modelPtr_;

    public:

        void setDefaultRule(const AbstractPrediction* prediction) override;

        void addRule(const ConditionList& conditions, const AbstractPrediction& prediction) override;

        std::unique_ptr<IModel> build() override;

};
