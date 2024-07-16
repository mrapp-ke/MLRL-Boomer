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
        }

        throw std::invalid_argument(
          "Invalid value given for parameter \"outputRatio\": Must be greater than 0 when used in regression problems, "
          "but is "
          + std::to_string(outputRatio));
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
            assertGreater<float32>("outputRatio", outputRatio, 0);
            assertLess<float32>("outputRatio", outputRatio, 1);
        }

        outputRatio_ = outputRatio;
        return *this;
    }

    uint32 FixedPartialHeadConfig::getMinOutputs() const {
        return minOutputs_;
    }

    IFixedPartialHeadConfig& FixedPartialHeadConfig::setMinOutputs(uint32 minOutputs) {
        assertGreaterOrEqual<uint32>("minOutputs", minOutputs, 2);
        minOutputs_ = minOutputs;
        return *this;
    }

    uint32 FixedPartialHeadConfig::getMaxOutputs() const {
        return maxOutputs_;
    }

    IFixedPartialHeadConfig& FixedPartialHeadConfig::setMaxOutputs(uint32 maxOutputs) {
        if (maxOutputs != 0) assertGreaterOrEqual<uint32>("maxOutputs", maxOutputs, minOutputs_);
        maxOutputs_ = maxOutputs;
        return *this;
    }

    std::unique_ptr<IClassificationStatisticsProviderFactory> FixedPartialHeadConfig::createStatisticsProviderFactory(
      const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix,
      const IDecomposableClassificationLossConfig& lossConfig) const {
        uint32 numThreads = multiThreadingConfig_.get().getNumThreads(featureMatrix, labelMatrix.getNumOutputs());
        float32 outputRatio = calculateOutputRatio(outputRatio_, labelMatrix);
        std::unique_ptr<IDecomposableClassificationLossFactory> lossFactoryPtr =
          lossConfig.createDecomposableClassificationLossFactory();
        std::unique_ptr<IEvaluationMeasureFactory> evaluationMeasureFactoryPtr =
          lossConfig.createEvaluationMeasureFactory();
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
          std::move(regularRuleEvaluationFactoryPtr), std::move(pruningRuleEvaluationFactoryPtr), numThreads);
    }

    std::unique_ptr<IClassificationStatisticsProviderFactory> FixedPartialHeadConfig::createStatisticsProviderFactory(
      const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix,
      const ISparseDecomposableClassificationLossConfig& lossConfig) const {
        uint32 numThreads = multiThreadingConfig_.get().getNumThreads(featureMatrix, labelMatrix.getNumOutputs());
        float32 outputRatio = calculateOutputRatio(outputRatio_, labelMatrix);
        std::unique_ptr<ISparseDecomposableClassificationLossFactory> lossFactoryPtr =
          lossConfig.createSparseDecomposableClassificationLossFactory();
        std::unique_ptr<ISparseEvaluationMeasureFactory> evaluationMeasureFactoryPtr =
          lossConfig.createSparseEvaluationMeasureFactory();
        std::unique_ptr<ISparseDecomposableRuleEvaluationFactory> regularRuleEvaluationFactoryPtr =
          labelBinningConfig_.get().createDecomposableFixedPartialRuleEvaluationFactory(outputRatio, minOutputs_,
                                                                                        maxOutputs_);
        std::unique_ptr<ISparseDecomposableRuleEvaluationFactory> pruningRuleEvaluationFactoryPtr =
          labelBinningConfig_.get().createDecomposableFixedPartialRuleEvaluationFactory(outputRatio, minOutputs_,
                                                                                          maxOutputs_);
        return std::make_unique<SparseDecomposableClassificationStatisticsProviderFactory>(
          std::move(lossFactoryPtr), std::move(evaluationMeasureFactoryPtr), std::move(regularRuleEvaluationFactoryPtr),
          std::move(pruningRuleEvaluationFactoryPtr), numThreads);
    }

    std::unique_ptr<IClassificationStatisticsProviderFactory> FixedPartialHeadConfig::createStatisticsProviderFactory(
      const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix,
      const INonDecomposableClassificationLossConfig& lossConfig, const Blas& blas, const Lapack& lapack) const {
        uint32 numThreads = multiThreadingConfig_.get().getNumThreads(featureMatrix, labelMatrix.getNumOutputs());
        float32 outputRatio = calculateOutputRatio(outputRatio_, labelMatrix);
        std::unique_ptr<INonDecomposableClassificationLossFactory> lossFactoryPtr =
          lossConfig.createNonDecomposableClassificationLossFactory();
        std::unique_ptr<IEvaluationMeasureFactory> evaluationMeasureFactoryPtr =
          lossConfig.createNonDecomposableClassificationLossFactory();
        std::unique_ptr<INonDecomposableRuleEvaluationFactory> defaultRuleEvaluationFactoryPtr =
          labelBinningConfig_.get().createNonDecomposableCompleteRuleEvaluationFactory(blas, lapack);
        std::unique_ptr<INonDecomposableRuleEvaluationFactory> regularRuleEvaluationFactoryPtr =
          labelBinningConfig_.get().createNonDecomposableFixedPartialRuleEvaluationFactory(outputRatio, minOutputs_,
                                                                                           maxOutputs_, blas, lapack);
        std::unique_ptr<INonDecomposableRuleEvaluationFactory> pruningRuleEvaluationFactoryPtr =
          labelBinningConfigr_.get().createNonDecomposableFixedPartialRuleEvaluationFactory(outputRatio, minOutputs_,
                                                                                             maxOutputs_, blas, lapack);
        return std::make_unique<DenseNonDecomposableClassificationStatisticsProviderFactory>(
          std::move(lossFactoryPtr), std::move(evaluationMeasureFactoryPtr), std::move(defaultRuleEvaluationFactoryPtr),
          std::move(regularRuleEvaluationFactoryPtr), std::move(pruningRuleEvaluationFactoryPtr), numThreads);
    }

    std::unique_ptr<IRegressionStatisticsProviderFactory> FixedPartialHeadConfig::createStatisticsProviderFactory(
      const IFeatureMatrix& featureMatrix, const IRowWiseRegressionMatrix& regressionMatrix,
      const IDecomposableRegressionLossConfig& lossConfig) const {
        uint32 numThreads =
          multiThreadingConfigGetter_().getNumThreads(featureMatrix, regressionMatrix.getNumOutputs());
        float32 outputRatio = calculateOutputRatio(outputRatio_, regressionMatrix);
        std::unique_ptr<IDecomposableRegressionLossFactory> lossFactoryPtr =
          lossConfig.createDecomposableRegressionLossFactory();
        std::unique_ptr<IEvaluationMeasureFactory> evaluationMeasureFactoryPtr =
          lossConfig.createEvaluationMeasureFactory();
        std::unique_ptr<IDecomposableRuleEvaluationFactory> defaultRuleEvaluationFactoryPtr =
          labelBinningConfigGetter_().createDecomposableCompleteRuleEvaluationFactory();
        std::unique_ptr<IDecomposableRuleEvaluationFactory> regularRuleEvaluationFactoryPtr =
          labelBinningConfigGetter_().createDecomposableFixedPartialRuleEvaluationFactory(outputRatio, minOutputs_,
                                                                                          maxOutputs_);
        std::unique_ptr<IDecomposableRuleEvaluationFactory> pruningRuleEvaluationFactoryPtr =
          labelBinningConfigGetter_().createDecomposableFixedPartialRuleEvaluationFactory(outputRatio, minOutputs_,
                                                                                          maxOutputs_);
        return std::make_unique<DenseDecomposableRegressionStatisticsProviderFactory>(
          std::move(lossFactoryPtr), std::move(evaluationMeasureFactoryPtr), std::move(defaultRuleEvaluationFactoryPtr),
          std::move(regularRuleEvaluationFactoryPtr), std::move(pruningRuleEvaluationFactoryPtr), numThreads);
    }

    std::unique_ptr<IRegressionStatisticsProviderFactory> FixedPartialHeadConfig::createStatisticsProviderFactory(
      const IFeatureMatrix& featureMatrix, const IRowWiseRegressionMatrix& regressionMatrix,
      const INonDecomposableRegressionLossConfig& lossConfig, const Blas& blas, const Lapack& lapack) const {
        uint32 numThreads =
          multiThreadingConfigGetter_().getNumThreads(featureMatrix, regressionMatrix.getNumOutputs());
        float32 outputRatio = calculateOutputRatio(outputRatio_, regressionMatrix);
        std::unique_ptr<INonDecomposableRegressionLossFactory> lossFactoryPtr =
          lossConfig.createNonDecomposableRegressionLossFactory();
        std::unique_ptr<IEvaluationMeasureFactory> evaluationMeasureFactoryPtr =
          lossConfig.createNonDecomposableRegressionLossFactory();
        std::unique_ptr<INonDecomposableRuleEvaluationFactory> defaultRuleEvaluationFactoryPtr =
          labelBinningConfigGetter_().createNonDecomposableCompleteRuleEvaluationFactory(blas, lapack);
        std::unique_ptr<INonDecomposableRuleEvaluationFactory> regularRuleEvaluationFactoryPtr =
          labelBinningConfigGetter_().createNonDecomposableFixedPartialRuleEvaluationFactory(outputRatio, minOutputs_,
                                                                                             maxOutputs_, blas, lapack);
        std::unique_ptr<INonDecomposableRuleEvaluationFactory> pruningRuleEvaluationFactoryPtr =
          labelBinningConfigGetter_().createNonDecomposableFixedPartialRuleEvaluationFactory(outputRatio, minOutputs_,
                                                                                             maxOutputs_, blas, lapack);
        return std::make_unique<DenseNonDecomposableRegressionStatisticsProviderFactory>(
          std::move(lossFactoryPtr), std::move(evaluationMeasureFactoryPtr), std::move(defaultRuleEvaluationFactoryPtr),
          std::move(regularRuleEvaluationFactoryPtr), std::move(pruningRuleEvaluationFactoryPtr), numThreads);
    }

    bool FixedPartialHeadConfig::isPartial() const {
        return true;
    }

    bool FixedPartialHeadConfig::isSingleOutput() const {
        return false;
    }

}
