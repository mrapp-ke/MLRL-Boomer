#include "mlrl/boosting/model/model_builder_hdf5.hpp"

#include "mlrl/common/model/rule_model_hdf5.hpp"

namespace boosting {

    /**
     * Allows to build models that store several rules in an in-memory HDF5 file.
     */
    class Hdf5RuleModelBuilder final : public IModelBuilder {
        public:

            /**
             * @see `IModelBuilder::setDefaultRule`
             */
            void setDefaultRule(std::unique_ptr<IEvaluatedPrediction>& predictionPtr) override {
                // TODO
            }

            /**
             * @see `IModelBuilder::addRule`
             */
            void addRule(std::unique_ptr<ConditionList>& conditionListPtr,
                         std::unique_ptr<IEvaluatedPrediction>& predictionPtr) override {
                // TODO
            }

            /**
             * @see `IModelBuilder::setNumUsedRules`
             */
            void setNumUsedRules(uint32 numUsedRules) override {
                // TODO
            }

            /**
             * @see `IModelBuilder::buildModel`
             */
            std::unique_ptr<IRuleModel> buildModel() override {
                return std::make_unique<Hdf5RuleModel>(true);
            }
    };

    std::unique_ptr<IModelBuilder> Hdf5RuleModelBuilderFactory::create() const {
        return std::make_unique<Hdf5RuleModelBuilder>();
    }

}
