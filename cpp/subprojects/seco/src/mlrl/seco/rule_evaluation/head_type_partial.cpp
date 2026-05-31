#include "mlrl/seco/rule_evaluation/head_type_partial.hpp"

#include "mlrl/common/math/vector_math.hpp"
#include "mlrl/common/simd/vector_math.hpp"
#include "mlrl/seco/rule_evaluation/rule_evaluation_decomposable_partial.hpp"
#include "mlrl/seco/statistics/statistics_provider_decomposable_sparse.hpp"
#include "rule_evaluation_decomposable_majority.hpp"

namespace seco {

    PartialHeadConfig::PartialHeadConfig(ReadableProperty<IHeuristicConfig> heuristicConfig,
                                         ReadableProperty<IHeuristicConfig> pruningHeuristicConfig,
                                         ReadableProperty<ILiftFunctionConfig> liftFunctionConfig,
                                         ReadableProperty<ISimdConfig> simdConfig)
        : heuristicConfig_(heuristicConfig), pruningHeuristicConfig_(pruningHeuristicConfig),
          liftFunctionConfig_(liftFunctionConfig), simdConfig_(simdConfig) {}

    std::unique_ptr<IClassificationStatisticsProviderFactory> PartialHeadConfig::createStatisticsProviderFactory(
      const IRowWiseLabelMatrix& labelMatrix) const {
        auto defaultRuleEvaluationFactoryPtr = std::make_unique<DecomposableMajorityRuleEvaluationFactory>();
        auto regularRuleEvaluationFactoryPtr = std::make_unique<DecomposablePartialRuleEvaluationFactory>(
          heuristicConfig_.get().createHeuristicFactory(),
          liftFunctionConfig_.get().createLiftFunctionFactory(labelMatrix));
        auto pruningRuleEvaluationFactoryPtr = std::make_unique<DecomposablePartialRuleEvaluationFactory>(
          pruningHeuristicConfig_.get().createHeuristicFactory(),
          liftFunctionConfig_.get().createLiftFunctionFactory(labelMatrix));

#if SIMD_SUPPORT_ENABLED
        if (simdConfig_.get().isSimdRecommended(labelMatrix.getNumOutputs())) {
            return std::make_unique<SparseDecomposableStatisticsProviderFactory<SimdVectorMath>>(
              std::move(defaultRuleEvaluationFactoryPtr), std::move(regularRuleEvaluationFactoryPtr),
              std::move(pruningRuleEvaluationFactoryPtr));
        }
#endif

        return std::make_unique<SparseDecomposableStatisticsProviderFactory<SequentialVectorMath>>(
          std::move(defaultRuleEvaluationFactoryPtr), std::move(regularRuleEvaluationFactoryPtr),
          std::move(pruningRuleEvaluationFactoryPtr));
    }

}
