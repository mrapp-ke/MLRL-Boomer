#include "mlrl/seco/learner_seco.hpp"

namespace seco {

    MultiLabelSeCoRuleLearner::Config::Config() {
        this->useSequentialRuleModelAssemblage();
        this->useGreedyTopDownRuleInduction();
        this->useDefaultRule();
        this->useNoLabelSampling();
        this->useNoInstanceSampling();
        this->useNoFeatureSampling();
        this->useNoPartitionSampling();
        this->useGreedyTopDownRuleInduction();
        this->useCoverageStoppingCriterion();
        this->useSizeStoppingCriterion();
        this->useNoTimeStoppingCriterion();
        this->useLabelWiseStratifiedInstanceSampling();
        this->useSingleLabelHeads();
        this->useIrepRulePruning();
        this->useNoSequentialPostOptimization();
        this->useFMeasureHeuristic();
        this->useAccuracyPruningHeuristic();
        this->usePeakLiftFunction();
        this->useParallelRuleRefinement();
        this->useNoParallelStatisticUpdate();
        this->useParallelPrediction();
        this->useLabelWiseBinaryPredictor();
    }

    IGreedyTopDownRuleInductionConfig& MultiLabelSeCoRuleLearner::Config::useGreedyTopDownRuleInduction() {
        IGreedyTopDownRuleInductionConfig& ref = IGreedyTopDownRuleInductionMixin::useGreedyTopDownRuleInduction();
        ref.setRecalculatePredictions(false);
        return ref;
    }

    IBeamSearchTopDownRuleInductionConfig& MultiLabelSeCoRuleLearner::Config::useBeamSearchTopDownRuleInduction() {
        IBeamSearchTopDownRuleInductionConfig& ref =
          IBeamSearchTopDownRuleInductionMixin::useBeamSearchTopDownRuleInduction();
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