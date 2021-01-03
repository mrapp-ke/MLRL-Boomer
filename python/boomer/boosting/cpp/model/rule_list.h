/**
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "../../../common/cpp/model/model_builder.h"

// Forward declarations
class RuleList;


/**
 * Allows to build models that store several rules in the order they have been added. For prediction, a linear
 * combination of the scores that are provided by the individual rules is computed, i.e., the prediction is invariant to
 * the order of the rules.
 */
class RuleListBuilder final : public IModelBuilder {

    private:

        std::unique_ptr<RuleList> modelPtr_;

    public:

        void setDefaultRule(const AbstractPrediction* prediction) override;

        void addRule(const ConditionList& conditions, const AbstractPrediction& prediction) override;

        std::unique_ptr<IModel> build() override;

};
