/**
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "../../../common/cpp/model/model_builder.h"

// Forward declarations
class DecisionList;
class Rule;

/**
 * Allows to build models that store several rules in a decision list.
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
