#include "boosting/rule_evaluation/head_type_single.hpp"
#include "boosting/rule_evaluation/rule_evaluation_label_wise_single.hpp"
#include "boosting/statistics/statistics_provider_example_wise_dense.hpp"
#include "boosting/statistics/statistics_provider_label_wise_dense.hpp"


namespace boosting {

    SingleLabelHeadConfig::SingleLabelHeadConfig(const std::unique_ptr<ILabelBinningConfig>& labelBinningConfigPtr)
        : labelBinningConfigPtr_(labelBinningConfigPtr) {

    }

    std::unique_ptr<IStatisticsProviderFactory> SingleLabelHeadConfig::configure(
            const ILabelWiseLossConfig& lossConfig) const {
        float64 l1RegularizationWeight = 0;  // TODO Use correct value
        float64 l2RegularizationWeight = 0;  // TODO Use correct value
        uint32 numThreads = 1;  // TODO Use correct value
        std::unique_ptr<ILabelWiseLossFactory> lossFactoryPtr = lossConfig.configureLabelWise();
        std::unique_ptr<IEvaluationMeasureFactory> evaluationMeasureFactoryPtr = lossConfig.configureLabelWise();
        std::unique_ptr<ILabelWiseRuleEvaluationFactory> defaultRuleEvaluationFactoryPtr =
            labelBinningConfigPtr_->configureLabelWise();
        std::unique_ptr<ILabelWiseRuleEvaluationFactory> regularRuleEvaluationFactoryPtr =
            std::make_unique<LabelWiseSingleLabelRuleEvaluationFactory>(l1RegularizationWeight, l2RegularizationWeight);
        std::unique_ptr<ILabelWiseRuleEvaluationFactory> pruningRuleEvaluationFactoryPtr =
            std::make_unique<LabelWiseSingleLabelRuleEvaluationFactory>(l1RegularizationWeight, l2RegularizationWeight);
        return std::make_unique<DenseLabelWiseStatisticsProviderFactory>(
            std::move(lossFactoryPtr), std::move(evaluationMeasureFactoryPtr),
            std::move(defaultRuleEvaluationFactoryPtr), std::move(regularRuleEvaluationFactoryPtr),
            std::move(pruningRuleEvaluationFactoryPtr), numThreads);
    }

    std::unique_ptr<IStatisticsProviderFactory> SingleLabelHeadConfig::configure(
            const IExampleWiseLossConfig& lossConfig) const {
        float64 l1RegularizationWeight = 0;  // TODO Use correct value
        float64 l2RegularizationWeight = 0;  // TODO Use correct value
        uint32 numThreads = 1;  // TODO Use correct value
        std::unique_ptr<IExampleWiseLossFactory> lossFactoryPtr = lossConfig.configureExampleWise();
        std::unique_ptr<IEvaluationMeasureFactory> evaluationMeasureFactoryPtr = lossConfig.configureExampleWise();
        std::unique_ptr<IExampleWiseRuleEvaluationFactory> defaultRuleEvaluationFactoryPtr =
            labelBinningConfigPtr_->configureExampleWise();
        std::unique_ptr<ILabelWiseRuleEvaluationFactory> regularRuleEvaluationFactoryPtr =
            std::make_unique<LabelWiseSingleLabelRuleEvaluationFactory>(l1RegularizationWeight, l2RegularizationWeight);
        std::unique_ptr<ILabelWiseRuleEvaluationFactory> pruningRuleEvaluationFactoryPtr =
            std::make_unique<LabelWiseSingleLabelRuleEvaluationFactory>(l1RegularizationWeight, l2RegularizationWeight);
        return std::make_unique<DenseConvertibleExampleWiseStatisticsProviderFactory>(
            std::move(lossFactoryPtr), std::move(evaluationMeasureFactoryPtr),
            std::move(defaultRuleEvaluationFactoryPtr), std::move(regularRuleEvaluationFactoryPtr),
            std::move(pruningRuleEvaluationFactoryPtr), numThreads);
    }

}
