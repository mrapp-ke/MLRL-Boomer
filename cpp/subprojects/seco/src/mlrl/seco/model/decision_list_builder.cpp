#include "mlrl/seco/model/decision_list_builder.hpp"

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

            DecisionListBuilder() : modelPtr_(std::make_unique<RuleList>(false)) {}

            /**
             * @see `IModelBuilder::setDefaultRule`
             */
            void setDefaultRule(std::unique_ptr<AbstractEvaluatedPrediction>& predictionPtr) override {
                defaultHeadPtr_ = predictionPtr->createHead();
            }

            /**
             * @see `IModelBuilder::addRule`
             */
            void addRule(std::unique_ptr<ConditionList>& conditionListPtr,
                         std::unique_ptr<AbstractEvaluatedPrediction>& predictionPtr) override {
                modelPtr_->addRule(conditionListPtr->createConjunctiveBody(), predictionPtr->createHead());
            }

            /**
             * @see `IModelBuilder::setNumUsedRules`
             */
            void setNumUsedRules(uint32 numUsedRules) override {
                modelPtr_->setNumUsedRules(numUsedRules);
            }

            /**
             * @see `IModelBuilder::buildModel`
             */
            std::unique_ptr<IRuleModel> buildModel() override {
                if (defaultHeadPtr_) {
                    modelPtr_->addDefaultRule(std::move(defaultHeadPtr_));
                }

                return std::move(modelPtr_);
            }
    };

    std::unique_ptr<IModelBuilder> DecisionListBuilderFactory::create() const {
        return std::make_unique<DecisionListBuilder>();
    }

}
