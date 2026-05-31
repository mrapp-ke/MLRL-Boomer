#include "mlrl/seco/rule_evaluation/head_type_single.hpp"

#include "mlrl/common/math/vector_math.hpp"
#include "mlrl/common/simd/vector_math.hpp"
#include "mlrl/seco/rule_evaluation/rule_evaluation_decomposable_single.hpp"
#include "mlrl/seco/statistics/statistics_provider_decomposable_sparse.hpp"
#include "rule_evaluation_decomposable_majority.hpp"

namespace seco {

    SingleOutputHeadConfig::SingleOutputHeadConfig(ReadableProperty<IHeuristicConfig> heuristicConfig,
                                                   ReadableProperty<IHeuristicConfig> pruningHeuristicConfig,
                                                   ReadableProperty<ISimdConfig> simdConfig)
        : heuristicConfig_(heuristicConfig), pruningHeuristicConfig_(pruningHeuristicConfig), simdConfig_(simdConfig) {}

    std::unique_ptr<IClassificationStatisticsProviderFactory> SingleOutputHeadConfig::createStatisticsProviderFactory(
      const IRowWiseLabelMatrix& labelMatrix) const {
        auto defaultRuleEvaluationFactoryPtr = std::make_unique<DecomposableMajorityRuleEvaluationFactory>();
        auto regularRuleEvaluationFactoryPtr = std::make_unique<DecomposableSingleOutputRuleEvaluationFactory>(
          heuristicConfig_.get().createHeuristicFactory());
        auto pruningRuleEvaluationFactoryPtr = std::make_unique<DecomposableSingleOutputRuleEvaluationFactory>(
          pruningHeuristicConfig_.get().createHeuristicFactory());

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
