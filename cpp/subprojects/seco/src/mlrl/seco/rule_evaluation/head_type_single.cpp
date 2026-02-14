#include "mlrl/seco/rule_evaluation/head_type_single.hpp"

#include "mlrl/common/math/vector_math.hpp"
#include "mlrl/common/util/xsimd.hpp"
#include "mlrl/seco/rule_evaluation/rule_evaluation_decomposable_single.hpp"
#include "mlrl/seco/statistics/statistics_provider_decomposable_dense.hpp"
#include "rule_evaluation_decomposable_majority.hpp"

namespace seco {

    SingleOutputHeadConfig::SingleOutputHeadConfig(ReadableProperty<IHeuristicConfig> heuristicConfig,
                                                   ReadableProperty<IHeuristicConfig> pruningHeuristicConfig,
                                                   ReadableProperty<ISimdConfig> simdConfig)
        : heuristicConfig_(heuristicConfig), pruningHeuristicConfig_(pruningHeuristicConfig), simdConfig_(simdConfig) {}

    std::unique_ptr<IClassificationStatisticsProviderFactory> SingleOutputHeadConfig::createStatisticsProviderFactory(
      const IRowWiseLabelMatrix& labelMatrix) const {
        std::unique_ptr<IDecomposableRuleEvaluationFactory> defaultRuleEvaluationFactoryPtr =
          std::make_unique<DecomposableMajorityRuleEvaluationFactory>();
        std::unique_ptr<IDecomposableRuleEvaluationFactory> regularRuleEvaluationFactoryPtr =
          std::make_unique<DecomposableSingleOutputRuleEvaluationFactory>(
            heuristicConfig_.get().createHeuristicFactory());
        std::unique_ptr<IDecomposableRuleEvaluationFactory> pruningRuleEvaluationFactoryPtr =
          std::make_unique<DecomposableSingleOutputRuleEvaluationFactory>(
            pruningHeuristicConfig_.get().createHeuristicFactory());

#if SIMD_SUPPORT_ENABLED
        if (labelMatrix.getNumOutputs() > 1 && simdConfig_.get().isSimdEnabled()) {
            return std::make_unique<DenseDecomposableStatisticsProviderFactory<SimdArrayOperations>>(
              std::move(defaultRuleEvaluationFactoryPtr), std::move(regularRuleEvaluationFactoryPtr),
              std::move(pruningRuleEvaluationFactoryPtr));
        }
#endif

        return std::make_unique<DenseDecomposableStatisticsProviderFactory<SequentialArrayOperations>>(
          std::move(defaultRuleEvaluationFactoryPtr), std::move(regularRuleEvaluationFactoryPtr),
          std::move(pruningRuleEvaluationFactoryPtr));
    }

}
