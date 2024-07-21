#include "mlrl/seco/rule_evaluation/head_type_single.hpp"

#include "mlrl/seco/rule_evaluation/rule_evaluation_decomposable_single.hpp"
#include "mlrl/seco/statistics/statistics_provider_decomposable_dense.hpp"
#include "rule_evaluation_decomposable_majority.hpp"

namespace seco {

    SingleOutputHeadConfig::SingleOutputHeadConfig(ReadableProperty<IHeuristicConfig> heuristicConfigGetter,
                                                   ReadableProperty<IHeuristicConfig> pruningHeuristicConfigGetter)
        : heuristicConfig_(heuristicConfigGetter), pruningHeuristicConfig_(pruningHeuristicConfigGetter) {}

    std::unique_ptr<IStatisticsProviderFactory> SingleOutputHeadConfig::createStatisticsProviderFactory(
      const IRowWiseLabelMatrix& labelMatrix) const {
        std::unique_ptr<IDecomposableRuleEvaluationFactory> defaultRuleEvaluationFactoryPtr =
          std::make_unique<DecomposableMajorityRuleEvaluationFactory>();
        std::unique_ptr<IDecomposableRuleEvaluationFactory> regularRuleEvaluationFactoryPtr =
          std::make_unique<DecomposableSingleOutputRuleEvaluationFactory>(
            heuristicConfig_.get().createHeuristicFactory());
        std::unique_ptr<IDecomposableRuleEvaluationFactory> pruningRuleEvaluationFactoryPtr =
          std::make_unique<DecomposableSingleOutputRuleEvaluationFactory>(
            pruningHeuristicConfig_.get().createHeuristicFactory());
        return std::make_unique<DenseDecomposableStatisticsProviderFactory>(std::move(defaultRuleEvaluationFactoryPtr),
                                                                            std::move(regularRuleEvaluationFactoryPtr),
                                                                            std::move(pruningRuleEvaluationFactoryPtr));
    }

}
