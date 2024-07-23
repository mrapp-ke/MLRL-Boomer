#include "mlrl/seco/rule_evaluation/head_type_partial.hpp"

#include "mlrl/seco/rule_evaluation/rule_evaluation_decomposable_partial.hpp"
#include "mlrl/seco/statistics/statistics_provider_decomposable_dense.hpp"
#include "rule_evaluation_decomposable_majority.hpp"

namespace seco {

    PartialHeadConfig::PartialHeadConfig(ReadableProperty<IHeuristicConfig> heuristicConfig,
                                         ReadableProperty<IHeuristicConfig> pruningHeuristicConfig,
                                         ReadableProperty<ILiftFunctionConfig> liftFunctionConfig)
        : heuristicConfig_(heuristicConfig), pruningHeuristicConfig_(pruningHeuristicConfig),
          liftFunctionConfig_(liftFunctionConfig) {}

    std::unique_ptr<IClassificationStatisticsProviderFactory> PartialHeadConfig::createStatisticsProviderFactory(
      const IRowWiseLabelMatrix& labelMatrix) const {
        std::unique_ptr<IDecomposableRuleEvaluationFactory> defaultRuleEvaluationFactoryPtr =
          std::make_unique<DecomposableMajorityRuleEvaluationFactory>();
        std::unique_ptr<IDecomposableRuleEvaluationFactory> regularRuleEvaluationFactoryPtr =
          std::make_unique<DecomposablePartialRuleEvaluationFactory>(
            heuristicConfig_.get().createHeuristicFactory(),
            liftFunctionConfig_.get().createLiftFunctionFactory(labelMatrix));
        std::unique_ptr<IDecomposableRuleEvaluationFactory> pruningRuleEvaluationFactoryPtr =
          std::make_unique<DecomposablePartialRuleEvaluationFactory>(
            pruningHeuristicConfig_.get().createHeuristicFactory(),
            liftFunctionConfig_.get().createLiftFunctionFactory(labelMatrix));
        return std::make_unique<DenseDecomposableStatisticsProviderFactory>(std::move(defaultRuleEvaluationFactoryPtr),
                                                                            std::move(regularRuleEvaluationFactoryPtr),
                                                                            std::move(pruningRuleEvaluationFactoryPtr));
    }

}
