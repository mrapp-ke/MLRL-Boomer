#include "mlrl/seco/rule_evaluation/head_type_partial.hpp"

#include "mlrl/seco/rule_evaluation/rule_evaluation_decomposable_partial.hpp"
#include "mlrl/seco/statistics/statistics_provider_decomposable_dense.hpp"
#include "rule_evaluation_decomposable_majority.hpp"

namespace seco {

    PartialHeadConfig::PartialHeadConfig(const std::unique_ptr<IHeuristicConfig>& heuristicConfigPtr,
                                         const std::unique_ptr<IHeuristicConfig>& pruningHeuristicConfigPtr,
                                         const std::unique_ptr<ILiftFunctionConfig>& liftFunctionConfigPtr)
        : heuristicConfigPtr_(heuristicConfigPtr), pruningHeuristicConfigPtr_(pruningHeuristicConfigPtr),
          liftFunctionConfigPtr_(liftFunctionConfigPtr) {}

    std::unique_ptr<IStatisticsProviderFactory> PartialHeadConfig::createStatisticsProviderFactory(
      const IRowWiseLabelMatrix& labelMatrix) const {
        std::unique_ptr<IDecomposableRuleEvaluationFactory> defaultRuleEvaluationFactoryPtr =
          std::make_unique<DecomposableMajorityRuleEvaluationFactory>();
        std::unique_ptr<IDecomposableRuleEvaluationFactory> regularRuleEvaluationFactoryPtr =
          std::make_unique<DecomposablePartialRuleEvaluationFactory>(
            heuristicConfigPtr_->createHeuristicFactory(),
            liftFunctionConfigPtr_->createLiftFunctionFactory(labelMatrix));
        std::unique_ptr<IDecomposableRuleEvaluationFactory> pruningRuleEvaluationFactoryPtr =
          std::make_unique<DecomposablePartialRuleEvaluationFactory>(
            pruningHeuristicConfigPtr_->createHeuristicFactory(),
            liftFunctionConfigPtr_->createLiftFunctionFactory(labelMatrix));
        return std::make_unique<DenseDecomposableStatisticsProviderFactory>(std::move(defaultRuleEvaluationFactoryPtr),
                                                                            std::move(regularRuleEvaluationFactoryPtr),
                                                                            std::move(pruningRuleEvaluationFactoryPtr));
    }

}
