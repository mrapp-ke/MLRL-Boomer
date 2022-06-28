#include "seco/learner.hpp"
#include "seco/heuristics/heuristic_precision.hpp"
#include "seco/lift_functions/lift_function_no.hpp"
#include "seco/model/decision_list_builder.hpp"
#include "seco/output/predictor_classification_label_wise.hpp"
#include "seco/rule_evaluation/head_type_single.hpp"
#include "common/output/label_space_info_no.hpp"


namespace seco {

    AbstractSeCoRuleLearner::Config::Config() {
        this->useNoCoverageStoppingCriterion();
        this->useSingleLabelHeads();
        this->useNoLiftFunction();
        this->usePrecisionHeuristic();
        this->usePrecisionPruningHeuristic();
        this->useLabelWiseClassificationPredictor();
    }

    const CoverageStoppingCriterionConfig* AbstractSeCoRuleLearner::Config::getCoverageStoppingCriterionConfig() const {
        return coverageStoppingCriterionConfigPtr_.get();
    }

    const IHeadConfig& AbstractSeCoRuleLearner::Config::getHeadConfig() const {
        return *headConfigPtr_;
    }

    const IHeuristicConfig& AbstractSeCoRuleLearner::Config::getHeuristicConfig() const {
        return *heuristicConfigPtr_;
    }

    const IHeuristicConfig& AbstractSeCoRuleLearner::Config::getPruningHeuristicConfig() const {
        return *pruningHeuristicConfigPtr_;
    }

    const ILiftFunctionConfig& AbstractSeCoRuleLearner::Config::getLiftFunctionConfig() const {
        return *liftFunctionConfigPtr_;
    }

    const IClassificationPredictorConfig& AbstractSeCoRuleLearner::Config::getClassificationPredictorConfig() const {
        return *classificationPredictorConfigPtr_;
    }

    IGreedyTopDownRuleInductionConfig& AbstractSeCoRuleLearner::Config::useGreedyTopDownRuleInduction() {
        IGreedyTopDownRuleInductionConfig& config = AbstractRuleLearner::Config::useGreedyTopDownRuleInduction();
        config.setRecalculatePredictions(false);
        return config;
    }

    void AbstractSeCoRuleLearner::Config::useNoCoverageStoppingCriterion() {
        coverageStoppingCriterionConfigPtr_ = nullptr;
    }

    void AbstractSeCoRuleLearner::Config::useSingleLabelHeads() {
        headConfigPtr_ = std::make_unique<SingleLabelHeadConfig>(heuristicConfigPtr_, pruningHeuristicConfigPtr_);
    }

    void AbstractSeCoRuleLearner::Config::useNoLiftFunction() {
        liftFunctionConfigPtr_ = std::make_unique<NoLiftFunctionConfig>();
    }

    void AbstractSeCoRuleLearner::Config::usePrecisionHeuristic() {
        heuristicConfigPtr_ = std::make_unique<PrecisionConfig>();
    }

    void AbstractSeCoRuleLearner::Config::usePrecisionPruningHeuristic() {
        pruningHeuristicConfigPtr_ = std::make_unique<PrecisionConfig>();
    }

    void AbstractSeCoRuleLearner::Config::useLabelWiseClassificationPredictor() {
        classificationPredictorConfigPtr_ =
            std::make_unique<LabelWiseClassificationPredictorConfig>(parallelPredictionConfigPtr_);
    }

    AbstractSeCoRuleLearner::AbstractSeCoRuleLearner(ISeCoRuleLearner::IConfig& config)
        : AbstractRuleLearner(config), config_(config) {

    }

    std::unique_ptr<IStoppingCriterionFactory> AbstractSeCoRuleLearner::createCoverageStoppingCriterionFactory() const {
        const CoverageStoppingCriterionConfig* config = config_.getCoverageStoppingCriterionConfig();
        return config ? config->createStoppingCriterionFactory() : nullptr;
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
        return config_.getHeadConfig().createStatisticsProviderFactory(labelMatrix);
    }

    std::unique_ptr<IModelBuilderFactory> AbstractSeCoRuleLearner::createModelBuilderFactory() const {
        return std::make_unique<DecisionListBuilderFactory>();
    }

    std::unique_ptr<ILabelSpaceInfo> AbstractSeCoRuleLearner::createLabelSpaceInfo(
            const IRowWiseLabelMatrix& labelMatrix) const {
        if (config_.getClassificationPredictorConfig().isLabelVectorSetNeeded()) {
            return createLabelVectorSet(labelMatrix);
        } else {
            return createNoLabelSpaceInfo();
        }
    }

    std::unique_ptr<IClassificationPredictorFactory> AbstractSeCoRuleLearner::createClassificationPredictorFactory(
            const IFeatureMatrix& featureMatrix, uint32 numLabels) const {
        return config_.getClassificationPredictorConfig().createClassificationPredictorFactory(featureMatrix,
                                                                                               numLabels);
    }

}
