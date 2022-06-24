#include "common/post_optimization/post_optimization_phase_list.hpp"


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

        IModelBuilder& getModelBuilder() const override {
            return *modelBuilderPtr_;
        }

        void optimizeModel(IThresholds& thresholds, const IRuleInduction& ruleInduction, IPartition& partition,
                           ILabelSampling& labelSampling, IInstanceSampling& instanceSampling,
                           IFeatureSampling& featureSampling, const IPruning& pruning,
                           const IPostProcessor& postProcessor, RNG& rng) const override {
            return;
        }

};

void PostOptimizationPhaseListFactory::addPostOptimizationPhaseFactory(
        std::unique_ptr<IPostOptimizationPhaseFactory> postOptimizationPhaseFactoryPtr) {
    postOptimizationPhaseFactories_.push_back(std::move(postOptimizationPhaseFactoryPtr));
}

std::unique_ptr<IPostOptimization> PostOptimizationPhaseListFactory::create(
        const IModelBuilderFactory& modelBuilderFactory) const {
    // TODO check if empty
    std::unique_ptr<IModelBuilder> modelBuilderPtr = modelBuilderFactory.create();
    return std::make_unique<NoPostOptimization>(std::move(modelBuilderPtr));
}
