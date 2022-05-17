#include "common/post_optimization/post_optimization_no.hpp"


/**
 * An implementation of the class `IPostOptimization` that does not perform any optimizations, but retains a previously
 * learned rule-based model.
 */
class NoPostOptimization final : public IPostOptimization {

    private:

        std::unique_ptr<IModelBuilder> modelBuilderPtr_;

    public:

        /**
         * @param modelBuilderPtr An unique pointer to an object of type `IModelBuilder` that should be used to build
         *                        the model
         */
        NoPostOptimization(std::unique_ptr<IModelBuilder> modelBuilderPtr)
            : modelBuilderPtr_(std::move(modelBuilderPtr)) {

        }

        void setDefaultRule(std::unique_ptr<AbstractEvaluatedPrediction>& predictionPtr) override {
            modelBuilderPtr_->setDefaultRule(predictionPtr);
        }

        void addRule(std::unique_ptr<ConditionList>& conditionListPtr,
                     std::unique_ptr<AbstractEvaluatedPrediction>& predictionPtr) override {
            modelBuilderPtr_->addRule(conditionListPtr, predictionPtr);
        }

        void optimizeModel(IThresholds& thresholds, const IRuleInduction& ruleInduction, IPartition& partition,
                           IInstanceSampling& instanceSampling, IFeatureSampling& featureSampling,
                           const IPruning& pruning, const IPostProcessor& postProcessor, RNG& rng) override {
            return;
        }

        std::unique_ptr<IRuleModel> buildModel(uint32 numUsedRules) override {
            return modelBuilderPtr_->buildModel(numUsedRules);
        }

};

/**
 * Allows to create instances of the type `IPostOptimization` that do not perform any optimizations, but retain a
 * previously learned rule-based model.
 */
class NoPostOptimizationFactory final : public IPostOptimizationFactory {

    public:

        std::unique_ptr<IPostOptimization> create(const IModelBuilderFactory& modelBuilderFactory) const override {
            std::unique_ptr<IModelBuilder> modelBuilderPtr = modelBuilderFactory.create();
            return std::make_unique<NoPostOptimization>(std::move(modelBuilderPtr));
        }

};

std::unique_ptr<IPostOptimizationFactory> NoPostOptimizationConfig::createPostOptimizationFactory() const {
    return std::make_unique<NoPostOptimizationFactory>();
}
