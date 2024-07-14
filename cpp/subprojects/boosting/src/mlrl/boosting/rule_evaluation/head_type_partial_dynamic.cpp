#include "mlrl/boosting/rule_evaluation/head_type_partial_dynamic.hpp"

#include "mlrl/boosting/statistics/statistics_provider_decomposable_dense.hpp"
#include "mlrl/boosting/statistics/statistics_provider_decomposable_sparse.hpp"
#include "mlrl/boosting/statistics/statistics_provider_non_decomposable_dense.hpp"
#include "mlrl/common/util/validation.hpp"

namespace boosting {

    DynamicPartialHeadConfig::DynamicPartialHeadConfig(
      ReadableProperty<ILabelBinningConfig> labelBinningConfigGetter,
      ReadableProperty<IMultiThreadingConfig> multiThreadingConfigGetter)
        : threshold_(0.02f), exponent_(2.0f), labelBinningConfig_(labelBinningConfigGetter),
          multiThreadingConfig_(multiThreadingConfigGetter) {}

    float32 DynamicPartialHeadConfig::getThreshold() const {
        return threshold_;
    }

    IDynamicPartialHeadConfig& DynamicPartialHeadConfig::setThreshold(float32 threshold) {
        assertGreater<float32>("threshold", threshold, 0);
        assertLess<float32>("threshold", threshold, 1);
        threshold_ = threshold;
        return *this;
    }

    float32 DynamicPartialHeadConfig::getExponent() const {
        return exponent_;
    }

    IDynamicPartialHeadConfig& DynamicPartialHeadConfig::setExponent(float32 exponent) {
        assertGreaterOrEqual<float32>("exponent", exponent, 1);
        exponent_ = exponent;
        return *this;
    }

    std::unique_ptr<IClassificationStatisticsProviderFactory> DynamicPartialHeadConfig::createStatisticsProviderFactory(
      const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix,
      const IDecomposableClassificationLossConfig& lossConfig) const {
        uint32 numThreads = multiThreadingConfig_.get().getNumThreads(featureMatrix, labelMatrix.getNumOutputs());
        std::unique_ptr<IDecomposableClassificationLossFactory> lossFactoryPtr =
          lossConfig.createDecomposableClassificationLossFactory();
        std::unique_ptr<IEvaluationMeasureFactory> evaluationMeasureFactoryPtr =
          lossConfig.createEvaluationMeasureFactory();
        std::unique_ptr<IDecomposableRuleEvaluationFactory> defaultRuleEvaluationFactoryPtr =
          labelBinningConfig_.get().createDecomposableCompleteRuleEvaluationFactory();
        std::unique_ptr<IDecomposableRuleEvaluationFactory> regularRuleEvaluationFactoryPtr =
          labelBinningConfig_.get().createDecomposableDynamicPartialRuleEvaluationFactory(threshold_, exponent_);
        std::unique_ptr<IDecomposableRuleEvaluationFactory> pruningRuleEvaluationFactoryPtr =
          labelBinningConfig_.get().createDecomposableDynamicPartialRuleEvaluationFactory(threshold_, exponent_);
        return std::make_unique<DenseDecomposableStatisticsProviderFactory>(
          std::move(lossFactoryPtr), std::move(evaluationMeasureFactoryPtr), std::move(defaultRuleEvaluationFactoryPtr),
          std::move(regularRuleEvaluationFactoryPtr), std::move(pruningRuleEvaluationFactoryPtr), numThreads);
    }

    std::unique_ptr<IClassificationStatisticsProviderFactory> DynamicPartialHeadConfig::createStatisticsProviderFactory(
      const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix,
      const ISparseDecomposableClassificationLossConfig& lossConfig) const {
        uint32 numThreads = multiThreadingConfig_.get().getNumThreads(featureMatrix, labelMatrix.getNumOutputs());
        std::unique_ptr<ISparseDecomposableClassificationLossFactory> lossFactoryPtr =
          lossConfig.createSparseDecomposableClassificationLossFactory();
        std::unique_ptr<ISparseEvaluationMeasureFactory> evaluationMeasureFactoryPtr =
          lossConfig.createSparseEvaluationMeasureFactory();
        std::unique_ptr<ISparseDecomposableRuleEvaluationFactory> regularRuleEvaluationFactoryPtr =
          labelBinningConfig_.get().createDecomposableDynamicPartialRuleEvaluationFactory(threshold_, exponent_);
        std::unique_ptr<ISparseDecomposableRuleEvaluationFactory> pruningRuleEvaluationFactoryPtr =
          labelBinningConfig_.get().createDecomposableDynamicPartialRuleEvaluationFactory(threshold_, exponent_);
        return std::make_unique<SparseDecomposableStatisticsProviderFactory>(
          std::move(lossFactoryPtr), std::move(evaluationMeasureFactoryPtr), std::move(regularRuleEvaluationFactoryPtr),
          std::move(pruningRuleEvaluationFactoryPtr), numThreads);
    }

    std::unique_ptr<IClassificationStatisticsProviderFactory> DynamicPartialHeadConfig::createStatisticsProviderFactory(
      const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix,
      const INonDecomposableClassificationLossConfig& lossConfig, const Blas& blas, const Lapack& lapack) const {
        uint32 numThreads = multiThreadingConfig_.get().getNumThreads(featureMatrix, labelMatrix.getNumOutputs());
        std::unique_ptr<INonDecomposableClassificationLossFactory> lossFactoryPtr =
          lossConfig.createNonDecomposableClassificationLossFactory();
        std::unique_ptr<IEvaluationMeasureFactory> evaluationMeasureFactoryPtr =
          lossConfig.createNonDecomposableClassificationLossFactory();
        std::unique_ptr<INonDecomposableRuleEvaluationFactory> defaultRuleEvaluationFactoryPtr =
          labelBinningConfig_.get().createNonDecomposableCompleteRuleEvaluationFactory(blas, lapack);
        std::unique_ptr<INonDecomposableRuleEvaluationFactory> regularRuleEvaluationFactoryPtr =
          labelBinningConfig_.get().createNonDecomposableDynamicPartialRuleEvaluationFactory(threshold_, exponent_,
                                                                                             blas, lapack);
        std::unique_ptr<INonDecomposableRuleEvaluationFactory> pruningRuleEvaluationFactoryPtr =
          labelBinningConfig_.get().createNonDecomposableDynamicPartialRuleEvaluationFactory(threshold_, exponent_,
                                                                                             blas, lapack);
        return std::make_unique<DenseNonDecomposableStatisticsProviderFactory>(
          std::move(lossFactoryPtr), std::move(evaluationMeasureFactoryPtr), std::move(defaultRuleEvaluationFactoryPtr),
          std::move(regularRuleEvaluationFactoryPtr), std::move(pruningRuleEvaluationFactoryPtr), numThreads);
    }

    bool DynamicPartialHeadConfig::isPartial() const {
        return true;
    }

    bool DynamicPartialHeadConfig::isSingleOutput() const {
        return false;
    }

}
