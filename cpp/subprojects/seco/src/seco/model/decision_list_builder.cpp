#include "seco/model/decision_list_builder.hpp"
#include "common/model/rule_list.hpp"


namespace seco {

    /**
     * Allows to build models that store several rules in the order they have been added, except for the default rule,
     * which is always located at the end.
     */
    class DecisionListBuilder final : public IModelBuilder {

        private:

            std::unique_ptr<IHead> defaultHeadPtr_;

            std::unique_ptr<RuleList> modelPtr_;

        public:

            DecisionListBuilder()
                : modelPtr_(std::make_unique<RuleList>()) {

            }

            void setDefaultRule(const AbstractPrediction& prediction) override {
                defaultHeadPtr_ = prediction.createHead();
            }

            void addRule(const ConditionList& conditions, const AbstractPrediction& prediction) override {
                modelPtr_->addRule(conditions.createConjunctiveBody(), prediction.createHead());
            }

            std::unique_ptr<IRuleModel> build(uint32 numUsedRules) override {
                if (defaultHeadPtr_) {
                    modelPtr_->addDefaultRule(std::move(defaultHeadPtr_));
                }

                modelPtr_->setNumUsedRules(numUsedRules);
                return std::move(modelPtr_);
            }

    };

    std::unique_ptr<IModelBuilder> DecisionListBuilderFactory::create() const {
        return std::make_unique<DecisionListBuilder>();
    }

}
