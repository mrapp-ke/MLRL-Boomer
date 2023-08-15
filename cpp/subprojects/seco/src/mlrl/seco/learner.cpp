#include "mlrl/seco/learner.hpp"

#include "mlrl/seco/model/decision_list_builder.hpp"
#include "mlrl/seco/rule_evaluation/rule_compare_function.hpp"

namespace seco {

    AbstractSeCoRuleLearner::Config::Config()
        : AbstractRuleLearner::Config(SECO_RULE_COMPARE_FUNCTION),
          headConfigPtr_(std::make_unique<SingleLabelHeadConfig>(heuristicConfigPtr_, pruningHeuristicConfigPtr_)),
          heuristicConfigPtr_(std::make_unique<PrecisionConfig>()),
          pruningHeuristicConfigPtr_(std::make_unique<PrecisionConfig>()),
          liftFunctionConfigPtr_(std::make_unique<NoLiftFunctionConfig>()) {}

    std::unique_ptr<CoverageStoppingCriterionConfig>&
      AbstractSeCoRuleLearner::Config::getCoverageStoppingCriterionConfigPtr() {
        return coverageStoppingCriterionConfigPtr_;
    }

    std::unique_ptr<IHeadConfig>& AbstractSeCoRuleLearner::Config::getHeadConfigPtr() {
        return headConfigPtr_;
    }

    std::unique_ptr<IHeuristicConfig>& AbstractSeCoRuleLearner::Config::getHeuristicConfigPtr() {
        return heuristicConfigPtr_;
    }

    std::unique_ptr<IHeuristicConfig>& AbstractSeCoRuleLearner::Config::getPruningHeuristicConfigPtr() {
        return pruningHeuristicConfigPtr_;
    }

    std::unique_ptr<ILiftFunctionConfig>& AbstractSeCoRuleLearner::Config::getLiftFunctionConfigPtr() {
        return liftFunctionConfigPtr_;
    }

    AbstractSeCoRuleLearner::AbstractSeCoRuleLearner(ISeCoRuleLearner::IConfig& config)
        : AbstractRuleLearner(config), config_(config) {}

    std::unique_ptr<IStoppingCriterionFactory> AbstractSeCoRuleLearner::createCoverageStoppingCriterionFactory() const {
        std::unique_ptr<CoverageStoppingCriterionConfig>& configPtr = config_.getCoverageStoppingCriterionConfigPtr();
        return configPtr ? configPtr->createStoppingCriterionFactory() : nullptr;
    }

    void AbstractSeCoRuleLearner::createStoppingCriterionFactories(StoppingCriterionListFactory& factory) const {
        AbstractRuleLearner::createStoppingCriterionFactories(factory);
        std::unique_ptr<IStoppingCriterionFactory> stoppingCriterionFactory =
          this->createCoverageStoppingCriterionFactory();

        if (stoppingCriterionFactory) {
            factory.addStoppingCriterionFactory(std::move(stoppingCriterionFactory));
        }
    }

    std::unique_ptr<IStatisticsProviderFactory> AbstractSeCoRuleLearner::createStatisticsProviderFactory(
      const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix) const {
        return config_.getHeadConfigPtr()->createStatisticsProviderFactory(labelMatrix);
    }

    std::unique_ptr<IModelBuilderFactory> AbstractSeCoRuleLearner::createModelBuilderFactory() const {
        return std::make_unique<DecisionListBuilderFactory>();
    }

    std::unique_ptr<ISparseBinaryPredictorFactory> AbstractSeCoRuleLearner::createSparseBinaryPredictorFactory(
      const IRowWiseFeatureMatrix& featureMatrix, uint32 numLabels) const {
        return config_.getBinaryPredictorConfigPtr()->createSparsePredictorFactory(featureMatrix, numLabels);
    }

}
