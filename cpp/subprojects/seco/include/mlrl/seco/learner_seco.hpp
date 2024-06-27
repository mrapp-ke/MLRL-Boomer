/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/seco/heuristics/heuristic_f_measure.hpp"
#include "mlrl/seco/heuristics/heuristic_m_estimate.hpp"
#include "mlrl/seco/learner.hpp"
#include "mlrl/seco/lift_functions/lift_function_kln.hpp"
#include "mlrl/seco/lift_functions/lift_function_peak.hpp"
#include "mlrl/seco/rule_evaluation/head_type.hpp"
#include "mlrl/seco/stopping/stopping_criterion_coverage.hpp"

#include <memory>

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
                            virtual public ISeCoRuleLearner::ISingleOutputHeadMixin,
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
                            virtual public ISeCoRuleLearner::IOutputWiseBinaryPredictionMixin,
                            virtual public ISequentialRuleModelAssemblageMixin,
                            virtual public IDefaultRuleMixin,
                            virtual public IGreedyTopDownRuleInductionMixin,
                            virtual public IBeamSearchTopDownRuleInductionMixin,
                            virtual public INoFeatureBinningMixin,
                            virtual public IEqualWidthFeatureBinningMixin,
                            virtual public IEqualFrequencyFeatureBinningMixin,
                            virtual public INoOutputSamplingMixin,
                            virtual public IRoundRobinOutputSamplingMixin,
                            virtual public IOutputSamplingWithoutReplacementMixin,
                            virtual public INoInstanceSamplingMixin,
                            virtual public IInstanceSamplingWithoutReplacementMixin,
                            virtual public IInstanceSamplingWithReplacementMixin,
                            virtual public IOutputWiseStratifiedInstanceSamplingMixin,
                            virtual public IExampleWiseStratifiedInstanceSamplingMixin,
                            virtual public INoFeatureSamplingMixin,
                            virtual public IFeatureSamplingWithoutReplacementMixin,
                            virtual public IRandomBiPartitionSamplingMixin,
                            virtual public INoPartitionSamplingMixin,
                            virtual public IOutputWiseStratifiedBiPartitionSamplingMixin,
                            virtual public IExampleWiseStratifiedBiPartitionSamplingMixin,
                            virtual public INoRulePruningMixin,
                            virtual public IIrepRulePruningMixin,
                            virtual public INoParallelRuleRefinementMixin,
                            virtual public IParallelRuleRefinementMixin,
                            virtual public INoParallelStatisticUpdateMixin,
                            virtual public IParallelStatisticUpdateMixin,
                            virtual public INoParallelPredictionMixin,
                            virtual public IParallelPredictionMixin,
                            virtual public INoSizeStoppingCriterionMixin,
                            virtual public ISizeStoppingCriterionMixin,
                            virtual public INoTimeStoppingCriterionMixin,
                            virtual public ITimeStoppingCriterionMixin,
                            virtual public INoSequentialPostOptimizationMixin,
                            virtual public ISequentialPostOptimizationMixin {
                public:

                    virtual ~IConfig() override {}
            };

            virtual ~IMultiLabelSeCoRuleLearner() override {}
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
