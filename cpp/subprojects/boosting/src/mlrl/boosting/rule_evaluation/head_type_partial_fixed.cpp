#include "mlrl/boosting/rule_evaluation/head_type_partial_fixed.hpp"

#include "mlrl/boosting/statistics/statistics_provider_decomposable_dense.hpp"
#include "mlrl/boosting/statistics/statistics_provider_decomposable_sparse.hpp"
#include "mlrl/boosting/statistics/statistics_provider_non_decomposable_dense.hpp"
#include "mlrl/common/util/validation.hpp"

namespace boosting {

    static inline float32 calculateOutputRatio(float32 outputRatio, const IRowWiseLabelMatrix& labelMatrix) {
        if (outputRatio > 0) {
            return outputRatio;
        } else {
            return labelMatrix.calculateLabelCardinality() / labelMatrix.getNumOutputs();
        }
    }

    static inline float32 calculateOutputRatio(float32 outputRatio, const IRowWiseRegressionMatrix& regressionMatrix) {
        if (outputRatio > 0) {
            return outputRatio;
        } else {
            return 0.33f;
        }
    }

    FixedPartialHeadConfig::FixedPartialHeadConfig(ReadableProperty<ILabelBinningConfig> labelBinningConfig,
                                                   ReadableProperty<IMultiThreadingConfig> multiThreadingConfig)
        : outputRatio_(0.0f), minOutputs_(2), maxOutputs_(0), labelBinningConfig_(labelBinningConfig),
          multiThreadingConfig_(multiThreadingConfig) {}

    float32 FixedPartialHeadConfig::getOutputRatio() const {
        return outputRatio_;
    }

    IFixedPartialHeadConfig& FixedPartialHeadConfig::setOutputRatio(float32 outputRatio) {
        if (!isEqualToZero(outputRatio)) {
            util::assertGreater<float32>("outputRatio", outputRatio, 0);
            util::assertLess<float32>("outputRatio", outputRatio, 1);
        }

        outputRatio_ = outputRatio;
        return *this;
    }

    uint32 FixedPartialHeadConfig::getMinOutputs() const {
        return minOutputs_;
    }

    IFixedPartialHeadConfig& FixedPartialHeadConfig::setMinOutputs(uint32 minOutputs) {
        util::assertGreaterOrEqual<uint32>("minOutputs", minOutputs, 2);
        minOutputs_ = minOutputs;
        return *this;
    }

    uint32 FixedPartialHeadConfig::getMaxOutputs() const {
        return maxOutputs_;
    }

    IFixedPartialHeadConfig& FixedPartialHeadConfig::setMaxOutputs(uint32 maxOutputs) {
        if (maxOutputs != 0) util::assertGreaterOrEqual<uint32>("maxOutputs", maxOutputs, minOutputs_);
        maxOutputs_ = maxOutputs;
        return *this;
    }

    std::unique_ptr<IClassificationStatisticsProviderFactory>
      FixedPartialHeadConfig::createClassificationStatisticsProviderFactory(
        const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix,
        const IDecomposableClassificationLossConfig& lossConfig) const {
        MultiThreadingSettings multiThreadingSettings =
          multiThreadingConfig_.get().getSettings(featureMatrix, labelMatrix.getNumOutputs());
        float32 outputRatio = calculateOutputRatio(outputRatio_, labelMatrix);
        std::unique_ptr<IDecomposableClassificationLossFactory<float64>> lossFactoryPtr =
          lossConfig.createDecomposableClassificationLossFactory();
        std::unique_ptr<IClassificationEvaluationMeasureFactory<float64>> evaluationMeasureFactoryPtr =
          lossConfig.createClassificationEvaluationMeasureFactory();
        std::unique_ptr<IDecomposableRuleEvaluationFactory> defaultRuleEvaluationFactoryPtr =
          labelBinningConfig_.get().createDecomposableCompleteRuleEvaluationFactory();
        std::unique_ptr<IDecomposableRuleEvaluationFactory> regularRuleEvaluationFactoryPtr =
          labelBinningConfig_.get().createDecomposableFixedPartialRuleEvaluationFactory(outputRatio, minOutputs_,
                                                                                        maxOutputs_);
        std::unique_ptr<IDecomposableRuleEvaluationFactory> pruningRuleEvaluationFactoryPtr =
          labelBinningConfig_.get().createDecomposableFixedPartialRuleEvaluationFactory(outputRatio, minOutputs_,
                                                                                        maxOutputs_);
        return std::make_unique<DenseDecomposableClassificationStatisticsProviderFactory>(
          std::move(lossFactoryPtr), std::move(evaluationMeasureFactoryPtr), std::move(defaultRuleEvaluationFactoryPtr),
          std::move(regularRuleEvaluationFactoryPtr), std::move(pruningRuleEvaluationFactoryPtr),
          multiThreadingSettings);
    }

    std::unique_ptr<IClassificationStatisticsProviderFactory>
      FixedPartialHeadConfig::createClassificationStatisticsProviderFactory(
        const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix,
        const ISparseDecomposableClassificationLossConfig& lossConfig) const {
        MultiThreadingSettings multiThreadingSettings =
          multiThreadingConfig_.get().getSettings(featureMatrix, labelMatrix.getNumOutputs());
        float32 outputRatio = calculateOutputRatio(outputRatio_, labelMatrix);
        std::unique_ptr<ISparseDecomposableClassificationLossFactory<float64>> lossFactoryPtr =
          lossConfig.createSparseDecomposableClassificationLossFactory();
        std::unique_ptr<ISparseEvaluationMeasureFactory<float64>> evaluationMeasureFactoryPtr =
          lossConfig.createSparseEvaluationMeasureFactory();
        std::unique_ptr<ISparseDecomposableRuleEvaluationFactory> regularRuleEvaluationFactoryPtr =
          labelBinningConfig_.get().createDecomposableFixedPartialRuleEvaluationFactory(outputRatio, minOutputs_,
                                                                                        maxOutputs_);
        std::unique_ptr<ISparseDecomposableRuleEvaluationFactory> pruningRuleEvaluationFactoryPtr =
          labelBinningConfig_.get().createDecomposableFixedPartialRuleEvaluationFactory(outputRatio, minOutputs_,
                                                                                        maxOutputs_);
        return std::make_unique<SparseDecomposableClassificationStatisticsProviderFactory>(
          std::move(lossFactoryPtr), std::move(evaluationMeasureFactoryPtr), std::move(regularRuleEvaluationFactoryPtr),
          std::move(pruningRuleEvaluationFactoryPtr), multiThreadingSettings);
    }

    std::unique_ptr<IClassificationStatisticsProviderFactory>
      FixedPartialHeadConfig::createClassificationStatisticsProviderFactory(
        const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix,
        const INonDecomposableClassificationLossConfig& lossConfig, const BlasFactory& blasFactory,
        const LapackFactory& lapackFactory) const {
        MultiThreadingSettings multiThreadingSettings =
          multiThreadingConfig_.get().getSettings(featureMatrix, labelMatrix.getNumOutputs());
        float32 outputRatio = calculateOutputRatio(outputRatio_, labelMatrix);
        std::unique_ptr<INonDecomposableClassificationLossFactory<float64>> lossFactoryPtr =
          lossConfig.createNonDecomposableClassificationLossFactory();
        std::unique_ptr<IClassificationEvaluationMeasureFactory<float64>> evaluationMeasureFactoryPtr =
          lossConfig.createClassificationEvaluationMeasureFactory();
        std::unique_ptr<INonDecomposableRuleEvaluationFactory> defaultRuleEvaluationFactoryPtr =
          labelBinningConfig_.get().createNonDecomposableCompleteRuleEvaluationFactory(blasFactory, lapackFactory);
        std::unique_ptr<INonDecomposableRuleEvaluationFactory> regularRuleEvaluationFactoryPtr =
          labelBinningConfig_.get().createNonDecomposableFixedPartialRuleEvaluationFactory(
            outputRatio, minOutputs_, maxOutputs_, blasFactory, lapackFactory);
        std::unique_ptr<INonDecomposableRuleEvaluationFactory> pruningRuleEvaluationFactoryPtr =
          labelBinningConfig_.get().createNonDecomposableFixedPartialRuleEvaluationFactory(
            outputRatio, minOutputs_, maxOutputs_, blasFactory, lapackFactory);
        return std::make_unique<DenseNonDecomposableClassificationStatisticsProviderFactory>(
          std::move(lossFactoryPtr), std::move(evaluationMeasureFactoryPtr), std::move(defaultRuleEvaluationFactoryPtr),
          std::move(regularRuleEvaluationFactoryPtr), std::move(pruningRuleEvaluationFactoryPtr),
          multiThreadingSettings);
    }

    std::unique_ptr<IRegressionStatisticsProviderFactory>
      FixedPartialHeadConfig::createRegressionStatisticsProviderFactory(
        const IFeatureMatrix& featureMatrix, const IRowWiseRegressionMatrix& regressionMatrix,
        const IDecomposableRegressionLossConfig& lossConfig) const {
        MultiThreadingSettings multiThreadingSettings =
          multiThreadingConfig_.get().getSettings(featureMatrix, regressionMatrix.getNumOutputs());
        float32 outputRatio = calculateOutputRatio(outputRatio_, regressionMatrix);
        std::unique_ptr<IDecomposableRegressionLossFactory<float64>> lossFactoryPtr =
          lossConfig.createDecomposableRegressionLossFactory();
        std::unique_ptr<IRegressionEvaluationMeasureFactory<float64>> evaluationMeasureFactoryPtr =
          lossConfig.createRegressionEvaluationMeasureFactory();
        std::unique_ptr<IDecomposableRuleEvaluationFactory> defaultRuleEvaluationFactoryPtr =
          labelBinningConfig_.get().createDecomposableCompleteRuleEvaluationFactory();
        std::unique_ptr<IDecomposableRuleEvaluationFactory> regularRuleEvaluationFactoryPtr =
          labelBinningConfig_.get().createDecomposableFixedPartialRuleEvaluationFactory(outputRatio, minOutputs_,
                                                                                        maxOutputs_);
        std::unique_ptr<IDecomposableRuleEvaluationFactory> pruningRuleEvaluationFactoryPtr =
          labelBinningConfig_.get().createDecomposableFixedPartialRuleEvaluationFactory(outputRatio, minOutputs_,
                                                                                        maxOutputs_);
        return std::make_unique<DenseDecomposableRegressionStatisticsProviderFactory>(
          std::move(lossFactoryPtr), std::move(evaluationMeasureFactoryPtr), std::move(defaultRuleEvaluationFactoryPtr),
          std::move(regularRuleEvaluationFactoryPtr), std::move(pruningRuleEvaluationFactoryPtr),
          multiThreadingSettings);
    }

    std::unique_ptr<IRegressionStatisticsProviderFactory>
      FixedPartialHeadConfig::createRegressionStatisticsProviderFactory(
        const IFeatureMatrix& featureMatrix, const IRowWiseRegressionMatrix& regressionMatrix,
        const INonDecomposableRegressionLossConfig& lossConfig, const BlasFactory& blasFactory,
        const LapackFactory& lapackFactory) const {
        MultiThreadingSettings multiThreadingSettings =
          multiThreadingConfig_.get().getSettings(featureMatrix, regressionMatrix.getNumOutputs());
        float32 outputRatio = calculateOutputRatio(outputRatio_, regressionMatrix);
        std::unique_ptr<INonDecomposableRegressionLossFactory> lossFactoryPtr =
          lossConfig.createNonDecomposableRegressionLossFactory();
        std::unique_ptr<IRegressionEvaluationMeasureFactory<float64>> evaluationMeasureFactoryPtr =
          lossConfig.createRegressionEvaluationMeasureFactory();
        std::unique_ptr<INonDecomposableRuleEvaluationFactory> defaultRuleEvaluationFactoryPtr =
          labelBinningConfig_.get().createNonDecomposableCompleteRuleEvaluationFactory(blasFactory, lapackFactory);
        std::unique_ptr<INonDecomposableRuleEvaluationFactory> regularRuleEvaluationFactoryPtr =
          labelBinningConfig_.get().createNonDecomposableFixedPartialRuleEvaluationFactory(
            outputRatio, minOutputs_, maxOutputs_, blasFactory, lapackFactory);
        std::unique_ptr<INonDecomposableRuleEvaluationFactory> pruningRuleEvaluationFactoryPtr =
          labelBinningConfig_.get().createNonDecomposableFixedPartialRuleEvaluationFactory(
            outputRatio, minOutputs_, maxOutputs_, blasFactory, lapackFactory);
        return std::make_unique<DenseNonDecomposableRegressionStatisticsProviderFactory>(
          std::move(lossFactoryPtr), std::move(evaluationMeasureFactoryPtr), std::move(defaultRuleEvaluationFactoryPtr),
          std::move(regularRuleEvaluationFactoryPtr), std::move(pruningRuleEvaluationFactoryPtr),
          multiThreadingSettings);
    }

    bool FixedPartialHeadConfig::isPartial() const {
        return true;
    }

    bool FixedPartialHeadConfig::isSingleOutput() const {
        return false;
    }

}
