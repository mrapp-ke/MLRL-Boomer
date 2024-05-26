#include "mlrl/seco/rule_evaluation/head_type_single.hpp"

#include "mlrl/seco/rule_evaluation/rule_evaluation_label_wise_single.hpp"
#include "mlrl/seco/statistics/statistics_provider_decomposable_dense.hpp"
#include "rule_evaluation_label_wise_majority.hpp"

namespace seco {

    SingleOutputHeadConfig::SingleOutputHeadConfig(const std::unique_ptr<IHeuristicConfig>& heuristicConfigPtr,
                                                   const std::unique_ptr<IHeuristicConfig>& pruningHeuristicConfigPtr)
        : heuristicConfigPtr_(heuristicConfigPtr), pruningHeuristicConfigPtr_(pruningHeuristicConfigPtr) {}

    std::unique_ptr<IStatisticsProviderFactory> SingleOutputHeadConfig::createStatisticsProviderFactory(
      const IRowWiseLabelMatrix& labelMatrix) const {
        std::unique_ptr<ILabelWiseRuleEvaluationFactory> defaultRuleEvaluationFactoryPtr =
          std::make_unique<LabelWiseMajorityRuleEvaluationFactory>();
        std::unique_ptr<ILabelWiseRuleEvaluationFactory> regularRuleEvaluationFactoryPtr =
          std::make_unique<LabelWiseSingleOutputRuleEvaluationFactory>(heuristicConfigPtr_->createHeuristicFactory());
        std::unique_ptr<ILabelWiseRuleEvaluationFactory> pruningRuleEvaluationFactoryPtr =
          std::make_unique<LabelWiseSingleOutputRuleEvaluationFactory>(
            pruningHeuristicConfigPtr_->createHeuristicFactory());
        return std::make_unique<DenseDecomposableStatisticsProviderFactory>(std::move(defaultRuleEvaluationFactoryPtr),
                                                                            std::move(regularRuleEvaluationFactoryPtr),
                                                                            std::move(pruningRuleEvaluationFactoryPtr));
    }

}
