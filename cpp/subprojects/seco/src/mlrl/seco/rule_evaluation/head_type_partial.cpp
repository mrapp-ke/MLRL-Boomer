#include "mlrl/seco/rule_evaluation/head_type_partial.hpp"

#include "mlrl/seco/rule_evaluation/rule_evaluation_label_wise_partial.hpp"
#include "mlrl/seco/statistics/statistics_provider_label_wise_dense.hpp"
#include "rule_evaluation_label_wise_majority.hpp"

namespace seco {

    PartialHeadConfig::PartialHeadConfig(const std::unique_ptr<IHeuristicConfig>& heuristicConfigPtr,
                                         const std::unique_ptr<IHeuristicConfig>& pruningHeuristicConfigPtr,
                                         const std::unique_ptr<ILiftFunctionConfig>& liftFunctionConfigPtr)
        : heuristicConfigPtr_(heuristicConfigPtr), pruningHeuristicConfigPtr_(pruningHeuristicConfigPtr),
          liftFunctionConfigPtr_(liftFunctionConfigPtr) {}

    std::unique_ptr<IStatisticsProviderFactory> PartialHeadConfig::createStatisticsProviderFactory(
      const IRowWiseLabelMatrix& labelMatrix) const {
        std::unique_ptr<ILabelWiseRuleEvaluationFactory> defaultRuleEvaluationFactoryPtr =
          std::make_unique<LabelWiseMajorityRuleEvaluationFactory>();
        std::unique_ptr<ILabelWiseRuleEvaluationFactory> regularRuleEvaluationFactoryPtr =
          std::make_unique<LabelWisePartialRuleEvaluationFactory>(
            heuristicConfigPtr_->createHeuristicFactory(),
            liftFunctionConfigPtr_->createLiftFunctionFactory(labelMatrix));
        std::unique_ptr<ILabelWiseRuleEvaluationFactory> pruningRuleEvaluationFactoryPtr =
          std::make_unique<LabelWisePartialRuleEvaluationFactory>(
            pruningHeuristicConfigPtr_->createHeuristicFactory(),
            liftFunctionConfigPtr_->createLiftFunctionFactory(labelMatrix));
        return std::make_unique<DenseLabelWiseStatisticsProviderFactory>(std::move(defaultRuleEvaluationFactoryPtr),
                                                                         std::move(regularRuleEvaluationFactoryPtr),
                                                                         std::move(pruningRuleEvaluationFactoryPtr));
    }

}
