/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#ifdef _WIN32
    #pragma warning(push)
    #pragma warning(disable : 4250)
#endif

#include "seco/heuristics/heuristic_f_measure.hpp"
#include "seco/heuristics/heuristic_m_estimate.hpp"
#include "seco/learner.hpp"
#include "seco/lift_functions/lift_function_kln.hpp"
#include "seco/lift_functions/lift_function_peak.hpp"
#include "seco/rule_evaluation/head_type.hpp"
#include "seco/stopping/stopping_criterion_coverage.hpp"

namespace seco {

    /**
     * Defines the interface of the multi-label SeCo algorithm.
     */
    class MLRLSECO_API IMultiLabelSeCoRuleLearner : virtual public ISeCoRuleLearner {
        public:

            /**
             * Defines an interface for all classes that allow to configure the multi-label SeCo algorithm.
             */
            class IConfig : virtual public ISeCoRuleLearner::IConfig,
                            virtual public ISeCoRuleLearner::ICoverageStoppingCriterionMixin,
                            virtual public ISeCoRuleLearner::IPartialHeadMixin,
                            virtual public ISeCoRuleLearner::IPeakLiftFunctionMixin,
                            virtual public ISeCoRuleLearner::IKlnLiftFunctionMixin,
                            virtual public ISeCoRuleLearner::IAccuracyHeuristicMixin,
                            virtual public ISeCoRuleLearner::IAccuracyPruningHeuristicMixin,
                            virtual public ISeCoRuleLearner::IFMeasureHeuristicMixin,
                            virtual public ISeCoRuleLearner::IFMeasurePruningHeuristicMixin,
                            virtual public ISeCoRuleLearner::IMEstimateHeuristicMixin,
                            virtual public ISeCoRuleLearner::IMEstimatePruningHeuristicMixin,
                            virtual public ISeCoRuleLearner::ILaplaceHeuristicMixin,
                            virtual public ISeCoRuleLearner::ILaplacePruningHeuristicMixin,
                            virtual public ISeCoRuleLearner::IRecallHeuristicMixin,
                            virtual public ISeCoRuleLearner::IRecallPruningHeuristicMixin,
                            virtual public ISeCoRuleLearner::IWraHeuristicMixin,
                            virtual public ISeCoRuleLearner::IWraPruningHeuristicMixin,
                            virtual public IRuleLearner::IDefaultRuleMixin,
                            virtual public IRuleLearner::IBeamSearchTopDownMixin,
                            virtual public IRuleLearner::INoFeatureBinningMixin,
                            virtual public IRuleLearner::INoLabelSamplingMixin,
                            virtual public IRuleLearner::ILabelSamplingWithoutReplacementMixin,
                            virtual public IRuleLearner::INoInstanceSamplingMixin,
                            virtual public IRuleLearner::IInstanceSamplingWithoutReplacementMixin,
                            virtual public IRuleLearner::IInstanceSamplingWithReplacementMixin,
                            virtual public IRuleLearner::ILabelWiseStratifiedInstanceSamplingMixin,
                            virtual public IRuleLearner::IExampleWiseStratifiedInstanceSamplingMixin,
                            virtual public IRuleLearner::INoFeatureSamplingMixin,
                            virtual public IRuleLearner::IFeatureSamplingWithoutReplacementMixin,
                            virtual public IRuleLearner::IRandomBiPartitionSamplingMixin,
                            virtual public IRuleLearner::ILabelWiseStratifiedBiPartitionSamplingMixin,
                            virtual public IRuleLearner::IExampleWiseStratifiedBiPartitionSamplingMixin,
                            virtual public IRuleLearner::IRulePruningMixin,
                            virtual public IRuleLearner::IParallelRuleRefinementMixin,
                            virtual public IRuleLearner::IParallelStatisticUpdateMixin,
                            virtual public IRuleLearner::IParallelPredictionMixin,
                            virtual public IRuleLearner::ISizeStoppingCriterionMixin,
                            virtual public IRuleLearner::ITimeStoppingCriterionMixin,
                            virtual public IRuleLearner::ISequentialPostOptimizationMixin {
                public:

                    virtual ~IConfig() override {};
            };

            virtual ~IMultiLabelSeCoRuleLearner() override {};
    };

    /**
     * The multi-label SeCo algorithm.
     */
    class MultiLabelSeCoRuleLearner final : public AbstractSeCoRuleLearner,
                                            virtual public IMultiLabelSeCoRuleLearner {
        public:

            /**
             * Allows to configure the multi-label SeCo algorithm.
             */
            class Config final : public AbstractSeCoRuleLearner::Config,
                                 virtual public IMultiLabelSeCoRuleLearner::IConfig {
                public:

                    Config();

                    /**
                     * @see `IRuleLearner::IConfig::useGreedyTopDownRuleInduction`
                     */
                    IGreedyTopDownRuleInductionConfig& useGreedyTopDownRuleInduction() override;

                    /**
                     * @see `IRuleLearner::IBeamSearchTopDownMixin::useBeamSearchTopDownRuleInduction`
                     */
                    IBeamSearchTopDownRuleInductionConfig& useBeamSearchTopDownRuleInduction() override;

                    /**
                     * @see `IRuleLearner::ISizeStoppingCriterionMixin::useSizeStoppingCriterion`
                     */
                    ISizeStoppingCriterionConfig& useSizeStoppingCriterion() override;
            };

        private:

            const std::unique_ptr<IMultiLabelSeCoRuleLearner::IConfig> configPtr_;

        public:

            /**
             * @param configPtr An unique pointer to an object of type `IMultiLabelSeCoRuleLearner::IConfig` that
             *                  specifies the configuration that should be used by the rule learner
             */
            MultiLabelSeCoRuleLearner(std::unique_ptr<IMultiLabelSeCoRuleLearner::IConfig> configPtr);
    };

    /**
     * Creates and returns a new object of type `IMultiLabelSeCoRuleLearner::IConfig`.
     *
     * @return An unique pointer to an object of type `IMultiLabelSeCoRuleLearner::IConfig` that has been created
     */
    MLRLSECO_API std::unique_ptr<IMultiLabelSeCoRuleLearner::IConfig> createMultiLabelSeCoRuleLearnerConfig();

    /**
     * Creates and returns a new object of type `IMultiLabelSeCoRuleLearner`.
     *
     * @param configPtr An unique pointer to an object of type `IMultiLabelSeCoRuleLearner::IConfig` that specifies the
     *                  configuration that should be used by the rule learner.
     * @return          An unique pointer to an object of type `IMultiLabelSeCoRuleLearner` that has been created
     */
    MLRLSECO_API std::unique_ptr<IMultiLabelSeCoRuleLearner> createMultiLabelSeCoRuleLearner(
      std::unique_ptr<IMultiLabelSeCoRuleLearner::IConfig> configPtr);

}

#ifdef _WIN32
    #pragma warning(pop)
#endif
