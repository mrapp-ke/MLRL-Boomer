#include "seco/learner_seco.hpp"

namespace seco {

    MultiLabelSeCoRuleLearner::Config::Config() {
        this->useDefaultRule();
        this->useNoFeatureBinning();
        this->useNoLabelSampling();
        this->useGreedyTopDownRuleInduction();
        this->useCoverageStoppingCriterion();
        this->useSizeStoppingCriterion();
        this->useLabelWiseStratifiedInstanceSampling();
        this->useIrepRulePruning();
        this->useFMeasureHeuristic();
        this->useAccuracyPruningHeuristic();
        this->usePeakLiftFunction();
        this->useParallelRuleRefinement();
        this->useParallelPrediction();
    }

    IGreedyTopDownRuleInductionConfig& MultiLabelSeCoRuleLearner::Config::useGreedyTopDownRuleInduction() {
        IGreedyTopDownRuleInductionConfig& ref = AbstractRuleLearner::Config::useGreedyTopDownRuleInduction();
        ref.setRecalculatePredictions(false);
        return ref;
    }

    IBeamSearchTopDownRuleInductionConfig& MultiLabelSeCoRuleLearner::Config::useBeamSearchTopDownRuleInduction() {
        IBeamSearchTopDownRuleInductionConfig& ref = IBeamSearchTopDownMixin::useBeamSearchTopDownRuleInduction();
        ref.setRecalculatePredictions(false);
        return ref;
    }

    ISizeStoppingCriterionConfig& MultiLabelSeCoRuleLearner::Config::useSizeStoppingCriterion() {
        ISizeStoppingCriterionConfig& ref = ISizeStoppingCriterionMixin::useSizeStoppingCriterion();
        ref.setMaxRules(500);
        return ref;
    }

    MultiLabelSeCoRuleLearner::MultiLabelSeCoRuleLearner(std::unique_ptr<IMultiLabelSeCoRuleLearner::IConfig> configPtr)
        : AbstractSeCoRuleLearner(*configPtr), configPtr_(std::move(configPtr)) {}

    std::unique_ptr<IMultiLabelSeCoRuleLearner::IConfig> createMultiLabelSeCoRuleLearnerConfig() {
        return std::make_unique<MultiLabelSeCoRuleLearner::Config>();
    }

    std::unique_ptr<IMultiLabelSeCoRuleLearner> createMultiLabelSeCoRuleLearner(
      std::unique_ptr<IMultiLabelSeCoRuleLearner::IConfig> configPtr) {
        return std::make_unique<MultiLabelSeCoRuleLearner>(std::move(configPtr));
    }

}
