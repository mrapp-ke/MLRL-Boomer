#include "boosting/model/rule_list_builder.hpp"
#include "common/model/rule_list.hpp"


namespace boosting {

    /**
     * Allows to build models that store several rules in the order they have been added.
     */
    class RuleListBuilder final : public IModelBuilder {

        private:

            std::unique_ptr<RuleList> modelPtr_;

        public:

            RuleListBuilder()
                : modelPtr_(std::make_unique<RuleList>()) {

            }

            void setDefaultRule(const AbstractPrediction& prediction) override {
                modelPtr_->addDefaultRule(prediction.createHead());
            }

            void addRule(const ConditionList& conditions, const AbstractPrediction& prediction) override {
                modelPtr_->addRule(conditions.createConjunctiveBody(), prediction.createHead());
            }

            std::unique_ptr<IRuleModel> build(uint32 numUsedRules) override {
                modelPtr_->setNumUsedRules(numUsedRules);
                return std::move(modelPtr_);
            }

    };

    std::unique_ptr<IModelBuilder> RuleListBuilderFactory::create() const {
        return std::make_unique<RuleListBuilder>();
    }

}
