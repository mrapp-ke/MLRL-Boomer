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
    class MLRLSECO_API ISeCoClassifier : virtual public IClassificationRuleLearner {
        public:

            /**
             * Defines an interface for all classes that allow to configure the multi-label SeCo algorithm.
             */
            class IConfig : virtual public ISeCoRuleLearnerConfig,
                            virtual public INoCoverageStoppingCriterionMixin,
                            virtual public ICoverageStoppingCriterionMixin,
                            virtual public ISingleOutputHeadMixin,
                            virtual public IPartialHeadMixin,
                            virtual public INoLiftFunctionMixin,
                            virtual public IPeakLiftFunctionMixin,
                            virtual public IKlnLiftFunctionMixin,
                            virtual public IAccuracyHeuristicMixin,
                            virtual public IAccuracyPruningHeuristicMixin,
                            virtual public IFMeasureHeuristicMixin,
                            virtual public IFMeasurePruningHeuristicMixin,
                            virtual public IMEstimateHeuristicMixin,
                            virtual public IMEstimatePruningHeuristicMixin,
                            virtual public ILaplaceHeuristicMixin,
                            virtual public ILaplacePruningHeuristicMixin,
                            virtual public IPrecisionHeuristicMixin,
                            virtual public IPrecisionPruningHeuristicMixin,
                            virtual public IRecallHeuristicMixin,
                            virtual public IRecallPruningHeuristicMixin,
                            virtual public IWraHeuristicMixin,
                            virtual public IWraPruningHeuristicMixin,
                            virtual public IOutputWiseBinaryPredictionMixin,
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

            virtual ~ISeCoClassifier() override {}
    };

    /**
     * Creates and returns a new object of type `ISeCoClassifier::IConfig`.
     *
     * @return An unique pointer to an object of type `ISeCoClassifier::IConfig` that has been created
     */
    MLRLSECO_API std::unique_ptr<ISeCoClassifier::IConfig> createSeCoClassifierConfig();

    /**
     * Creates and returns a new object of type `ISeCoClassifier`.
     *
     * @param configPtr An unique pointer to an object of type `ISeCoClassifier::IConfig` that specifies the
     *                  configuration that should be used by the rule learner.
     * @return          An unique pointer to an object of type `ISeCoClassifier` that has been created
     */
    MLRLSECO_API std::unique_ptr<ISeCoClassifier> createSeCoClassifier(
      std::unique_ptr<ISeCoClassifier::IConfig> configPtr);

}
