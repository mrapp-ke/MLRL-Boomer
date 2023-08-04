/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#ifdef _WIN32
    #pragma warning(push)
    #pragma warning(disable : 4250)
#endif

#include "mlrl/seco/heuristics/heuristic_f_measure.hpp"
#include "mlrl/seco/heuristics/heuristic_m_estimate.hpp"
#include "mlrl/seco/learner.hpp"
#include "mlrl/seco/lift_functions/lift_function_kln.hpp"
#include "mlrl/seco/lift_functions/lift_function_peak.hpp"
#include "mlrl/seco/rule_evaluation/head_type.hpp"
#include "mlrl/seco/stopping/stopping_criterion_coverage.hpp"

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
                            virtual public ISeCoRuleLearner::INoCoverageStoppingCriterionMixin,
                            virtual public ISeCoRuleLearner::ICoverageStoppingCriterionMixin,
                            virtual public ISeCoRuleLearner::ISingleLabelHeadMixin,
                            virtual public ISeCoRuleLearner::IPartialHeadMixin,
                            virtual public ISeCoRuleLearner::INoLiftFunctionMixin,
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
                            virtual public ISeCoRuleLearner::IPrecisionHeuristicMixin,
                            virtual public ISeCoRuleLearner::IPrecisionPruningHeuristicMixin,
                            virtual public ISeCoRuleLearner::IRecallHeuristicMixin,
                            virtual public ISeCoRuleLearner::IRecallPruningHeuristicMixin,
                            virtual public ISeCoRuleLearner::IWraHeuristicMixin,
                            virtual public ISeCoRuleLearner::IWraPruningHeuristicMixin,
                            virtual public ISeCoRuleLearner::ILabelWiseBinaryPredictionMixin,
                            virtual public IRuleLearner::ISequentialRuleModelAssemblageMixin,
                            virtual public IRuleLearner::IDefaultRuleMixin,
                            virtual public IRuleLearner::IGreedyTopDownRuleInductionMixin,
                            virtual public IRuleLearner::IBeamSearchTopDownRuleInductionMixin,
                            virtual public IRuleLearner::INoLabelSamplingMixin,
                            virtual public IRuleLearner::IRoundRobinLabelSamplingMixin,
                            virtual public IRuleLearner::ILabelSamplingWithoutReplacementMixin,
                            virtual public IRuleLearner::INoInstanceSamplingMixin,
                            virtual public IRuleLearner::IInstanceSamplingWithoutReplacementMixin,
                            virtual public IRuleLearner::IInstanceSamplingWithReplacementMixin,
                            virtual public IRuleLearner::ILabelWiseStratifiedInstanceSamplingMixin,
                            virtual public IRuleLearner::IExampleWiseStratifiedInstanceSamplingMixin,
                            virtual public IRuleLearner::INoFeatureSamplingMixin,
                            virtual public IRuleLearner::IFeatureSamplingWithoutReplacementMixin,
                            virtual public IRuleLearner::IRandomBiPartitionSamplingMixin,
                            virtual public IRuleLearner::INoPartitionSamplingMixin,
                            virtual public IRuleLearner::ILabelWiseStratifiedBiPartitionSamplingMixin,
                            virtual public IRuleLearner::IExampleWiseStratifiedBiPartitionSamplingMixin,
                            virtual public IRuleLearner::INoRulePruningMixin,
                            virtual public IRuleLearner::IIrepRulePruningMixin,
                            virtual public IRuleLearner::INoParallelRuleRefinementMixin,
                            virtual public IRuleLearner::IParallelRuleRefinementMixin,
                            virtual public IRuleLearner::INoParallelStatisticUpdateMixin,
                            virtual public IRuleLearner::IParallelStatisticUpdateMixin,
                            virtual public IRuleLearner::INoParallelPredictionMixin,
                            virtual public IRuleLearner::IParallelPredictionMixin,
                            virtual public IRuleLearner::INoSizeStoppingCriterionMixin,
                            virtual public IRuleLearner::ISizeStoppingCriterionMixin,
                            virtual public IRuleLearner::INoTimeStoppingCriterionMixin,
                            virtual public IRuleLearner::ITimeStoppingCriterionMixin,
                            virtual public IRuleLearner::INoSequentialPostOptimizationMixin,
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
                     * @see `IRuleLearner::IGreedyTopDownRuleInductionMixin::useGreedyTopDownRuleInduction`
                     */
                    IGreedyTopDownRuleInductionConfig& useGreedyTopDownRuleInduction() override;

                    /**
                     * @see `IRuleLearner::IBeamSearchTopDownRuleInductionMixin::useBeamSearchTopDownRuleInduction`
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
