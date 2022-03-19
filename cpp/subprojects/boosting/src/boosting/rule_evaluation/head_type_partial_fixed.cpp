#include "boosting/rule_evaluation/head_type_partial_fixed.hpp"
#include "boosting/rule_evaluation/rule_evaluation_label_wise_partial_fixed.hpp"
#include "boosting/statistics/statistics_provider_example_wise_dense.hpp"
#include "boosting/statistics/statistics_provider_label_wise_dense.hpp"
#include "common/util/validation.hpp"


namespace boosting {

    FixedPartialHeadConfig::FixedPartialHeadConfig(
            const std::unique_ptr<ILabelBinningConfig>& labelBinningConfigPtr,
            const std::unique_ptr<IMultiThreadingConfig>& multiThreadingConfigPtr,
            const std::unique_ptr<IRegularizationConfig>& l1RegularizationConfigPtr,
            const std::unique_ptr<IRegularizationConfig>& l2RegularizationConfigPtr)
        : labelRatio_(0.0f), minLabels_(2), maxLabels_(0), labelBinningConfigPtr_(labelBinningConfigPtr),
          multiThreadingConfigPtr_(multiThreadingConfigPtr), l1RegularizationConfigPtr_(l1RegularizationConfigPtr),
          l2RegularizationConfigPtr_(l2RegularizationConfigPtr) {

    }

    float32 FixedPartialHeadConfig::getLabelRatio() const {
        return labelRatio_;
    }

    IFixedPartialHeadConfig& FixedPartialHeadConfig::setLabelRatio(float32 labelRatio) {
        if (labelRatio != 0) {
            assertGreater<float32>("labelRatio", labelRatio, 0);
            assertLess<float32>("labelRatio", labelRatio, 1);
        }
        labelRatio_ = labelRatio;
        return *this;
    }

    uint32 FixedPartialHeadConfig::getMinLabels() const {
        return minLabels_;
    }

    IFixedPartialHeadConfig& FixedPartialHeadConfig::setMinLabels(uint32 minLabels) {
        assertGreaterOrEqual<uint32>("minLabels", minLabels, 2);
        minLabels_ = minLabels;
        return *this;
    }

    uint32 FixedPartialHeadConfig::getMaxLabels() const {
        return maxLabels_;
    }

    IFixedPartialHeadConfig& FixedPartialHeadConfig::setMaxLabels(uint32 maxLabels) {
        if (maxLabels != 0) { assertGreaterOrEqual<uint32>("maxLabels", maxLabels, minLabels_); }
        maxLabels_ = maxLabels;
        return *this;
    }

    std::unique_ptr<IStatisticsProviderFactory> FixedPartialHeadConfig::createStatisticsProviderFactory(
            const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix,
            const ILabelWiseLossConfig& lossConfig) const {
        uint32 numThreads = multiThreadingConfigPtr_->getNumThreads(featureMatrix, labelMatrix.getNumCols());
        float32 labelRatio =
            labelRatio_ != 0 ? labelRatio_ : labelMatrix.calculateLabelCardinality() / labelMatrix.getNumCols();
        std::unique_ptr<ILabelWiseLossFactory> lossFactoryPtr = lossConfig.createLabelWiseLossFactory();
        std::unique_ptr<IEvaluationMeasureFactory> evaluationMeasureFactoryPtr =
            lossConfig.createEvaluationMeasureFactory();
        std::unique_ptr<ILabelWiseRuleEvaluationFactory> defaultRuleEvaluationFactoryPtr =
            labelBinningConfigPtr_->createLabelWiseCompleteRuleEvaluationFactory();
        std::unique_ptr<ILabelWiseRuleEvaluationFactory> regularRuleEvaluationFactoryPtr =
            labelBinningConfigPtr_->createLabelWiseFixedPartialRuleEvaluationFactory(
                labelRatio, minLabels_, maxLabels_);
        std::unique_ptr<ILabelWiseRuleEvaluationFactory> pruningRuleEvaluationFactoryPtr =
            labelBinningConfigPtr_->createLabelWiseFixedPartialRuleEvaluationFactory(
                labelRatio, minLabels_, maxLabels_);
        return std::make_unique<DenseLabelWiseStatisticsProviderFactory>(
            std::move(lossFactoryPtr), std::move(evaluationMeasureFactoryPtr),
            std::move(defaultRuleEvaluationFactoryPtr), std::move(regularRuleEvaluationFactoryPtr),
            std::move(pruningRuleEvaluationFactoryPtr), numThreads);
    }

    std::unique_ptr<IStatisticsProviderFactory> FixedPartialHeadConfig::createStatisticsProviderFactory(
            const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix,
            const IExampleWiseLossConfig& lossConfig, const Blas& blas, const Lapack& lapack) const {
        uint32 numThreads = multiThreadingConfigPtr_->getNumThreads(featureMatrix, labelMatrix.getNumCols());
        std::unique_ptr<IExampleWiseLossFactory> lossFactoryPtr = lossConfig.createExampleWiseLossFactory();
        std::unique_ptr<IEvaluationMeasureFactory> evaluationMeasureFactoryPtr =
            lossConfig.createExampleWiseLossFactory();
        std::unique_ptr<IExampleWiseRuleEvaluationFactory> defaultRuleEvaluationFactoryPtr =
            labelBinningConfigPtr_->createExampleWiseCompleteRuleEvaluationFactory(blas, lapack);
        std::unique_ptr<IExampleWiseRuleEvaluationFactory> regularRuleEvaluationFactoryPtr =
            labelBinningConfigPtr_->createExampleWiseCompleteRuleEvaluationFactory(blas, lapack);
        std::unique_ptr<IExampleWiseRuleEvaluationFactory> pruningRuleEvaluationFactoryPtr =
            labelBinningConfigPtr_->createExampleWiseCompleteRuleEvaluationFactory(blas, lapack);
        return std::make_unique<DenseExampleWiseStatisticsProviderFactory>(
            std::move(lossFactoryPtr), std::move(evaluationMeasureFactoryPtr),
            std::move(defaultRuleEvaluationFactoryPtr), std::move(regularRuleEvaluationFactoryPtr),
            std::move(pruningRuleEvaluationFactoryPtr), numThreads);
    }

}
