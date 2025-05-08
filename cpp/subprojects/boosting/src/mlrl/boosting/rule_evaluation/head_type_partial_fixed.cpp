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

    template<typename StatisticType>
    class PartialFixedHeadPreset final : public IHeadConfig::IPreset<StatisticType> {
        private:

            ReadableProperty<ILabelBinningConfig> labelBinningConfig_;

            ReadableProperty<IMultiThreadingConfig> multiThreadingConfig_;

            float32 outputRatio_;

            uint32 minOutputs_;

            uint32 maxOutputs_;

        public:

            PartialFixedHeadPreset(ReadableProperty<ILabelBinningConfig> labelBinningConfig,
                                   ReadableProperty<IMultiThreadingConfig> multiThreadingConfig, float32 outputRatio,
                                   uint32 minOutputs, uint32 maxOutputs)
                : labelBinningConfig_(labelBinningConfig), multiThreadingConfig_(multiThreadingConfig),
                  outputRatio_(outputRatio), minOutputs_(minOutputs), maxOutputs_(maxOutputs) {}

            std::unique_ptr<IClassificationStatisticsProviderFactory> createClassificationStatisticsProviderFactory(
              const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix,
              std::unique_ptr<IDecomposableClassificationLossFactory<StatisticType>>& lossFactoryPtr,
              std::unique_ptr<IClassificationEvaluationMeasureFactory<StatisticType>>& evaluationMeasureFactoryPtr)
              const override {
                MultiThreadingSettings multiThreadingSettings =
                  multiThreadingConfig_.get().getSettings(featureMatrix, labelMatrix.getNumOutputs());
                float32 outputRatio = calculateOutputRatio(outputRatio_, labelMatrix);
                std::unique_ptr<IDecomposableRuleEvaluationFactory> defaultRuleEvaluationFactoryPtr =
                  labelBinningConfig_.get().createDecomposableCompleteRuleEvaluationFactory();
                std::unique_ptr<IDecomposableRuleEvaluationFactory> regularRuleEvaluationFactoryPtr =
                  labelBinningConfig_.get().createDecomposableFixedPartialRuleEvaluationFactory(
                    outputRatio, minOutputs_, maxOutputs_);
                std::unique_ptr<IDecomposableRuleEvaluationFactory> pruningRuleEvaluationFactoryPtr =
                  labelBinningConfig_.get().createDecomposableFixedPartialRuleEvaluationFactory(
                    outputRatio, minOutputs_, maxOutputs_);
                return std::make_unique<DenseDecomposableClassificationStatisticsProviderFactory<StatisticType>>(
                  std::move(lossFactoryPtr), std::move(evaluationMeasureFactoryPtr),
                  std::move(defaultRuleEvaluationFactoryPtr), std::move(regularRuleEvaluationFactoryPtr),
                  std::move(pruningRuleEvaluationFactoryPtr), multiThreadingSettings);
            }

            std::unique_ptr<IClassificationStatisticsProviderFactory> createClassificationStatisticsProviderFactory(
              const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix,
              std::unique_ptr<ISparseDecomposableClassificationLossFactory<StatisticType>>& lossFactoryPtr,
              std::unique_ptr<ISparseEvaluationMeasureFactory<StatisticType>>& evaluationMeasureFactoryPtr)
              const override {
                MultiThreadingSettings multiThreadingSettings =
                  multiThreadingConfig_.get().getSettings(featureMatrix, labelMatrix.getNumOutputs());
                float32 outputRatio = calculateOutputRatio(outputRatio_, labelMatrix);
                std::unique_ptr<ISparseDecomposableRuleEvaluationFactory> regularRuleEvaluationFactoryPtr =
                  labelBinningConfig_.get().createDecomposableFixedPartialRuleEvaluationFactory(
                    outputRatio, minOutputs_, maxOutputs_);
                std::unique_ptr<ISparseDecomposableRuleEvaluationFactory> pruningRuleEvaluationFactoryPtr =
                  labelBinningConfig_.get().createDecomposableFixedPartialRuleEvaluationFactory(
                    outputRatio, minOutputs_, maxOutputs_);
                return std::make_unique<SparseDecomposableClassificationStatisticsProviderFactory<StatisticType>>(
                  std::move(lossFactoryPtr), std::move(evaluationMeasureFactoryPtr),
                  std::move(regularRuleEvaluationFactoryPtr), std::move(pruningRuleEvaluationFactoryPtr),
                  multiThreadingSettings);
            }

            std::unique_ptr<IClassificationStatisticsProviderFactory> createClassificationStatisticsProviderFactory(
              const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix,
              std::unique_ptr<INonDecomposableClassificationLossFactory<StatisticType>>& lossFactoryPtr,
              std::unique_ptr<IClassificationEvaluationMeasureFactory<StatisticType>>& evaluationMeasureFactoryPtr,
              const BlasFactory& blasFactory, const LapackFactory& lapackFactory) const override {
                MultiThreadingSettings multiThreadingSettings =
                  multiThreadingConfig_.get().getSettings(featureMatrix, labelMatrix.getNumOutputs());
                float32 outputRatio = calculateOutputRatio(outputRatio_, labelMatrix);
                std::unique_ptr<INonDecomposableRuleEvaluationFactory> defaultRuleEvaluationFactoryPtr =
                  labelBinningConfig_.get().createNonDecomposableCompleteRuleEvaluationFactory(blasFactory,
                                                                                               lapackFactory);
                std::unique_ptr<INonDecomposableRuleEvaluationFactory> regularRuleEvaluationFactoryPtr =
                  labelBinningConfig_.get().createNonDecomposableFixedPartialRuleEvaluationFactory(
                    outputRatio, minOutputs_, maxOutputs_, blasFactory, lapackFactory);
                std::unique_ptr<INonDecomposableRuleEvaluationFactory> pruningRuleEvaluationFactoryPtr =
                  labelBinningConfig_.get().createNonDecomposableFixedPartialRuleEvaluationFactory(
                    outputRatio, minOutputs_, maxOutputs_, blasFactory, lapackFactory);
                return std::make_unique<DenseNonDecomposableClassificationStatisticsProviderFactory<StatisticType>>(
                  std::move(lossFactoryPtr), std::move(evaluationMeasureFactoryPtr),
                  std::move(defaultRuleEvaluationFactoryPtr), std::move(regularRuleEvaluationFactoryPtr),
                  std::move(pruningRuleEvaluationFactoryPtr), multiThreadingSettings);
            }

            std::unique_ptr<IRegressionStatisticsProviderFactory> createRegressionStatisticsProviderFactory(
              const IFeatureMatrix& featureMatrix, const IRowWiseRegressionMatrix& regressionMatrix,
              std::unique_ptr<IDecomposableRegressionLossFactory<StatisticType>>& lossFactoryPtr,
              std::unique_ptr<IRegressionEvaluationMeasureFactory<StatisticType>>& evaluationMeasureFactoryPtr)
              const override {
                MultiThreadingSettings multiThreadingSettings =
                  multiThreadingConfig_.get().getSettings(featureMatrix, regressionMatrix.getNumOutputs());
                float32 outputRatio = calculateOutputRatio(outputRatio_, regressionMatrix);
                std::unique_ptr<IDecomposableRuleEvaluationFactory> defaultRuleEvaluationFactoryPtr =
                  labelBinningConfig_.get().createDecomposableCompleteRuleEvaluationFactory();
                std::unique_ptr<IDecomposableRuleEvaluationFactory> regularRuleEvaluationFactoryPtr =
                  labelBinningConfig_.get().createDecomposableFixedPartialRuleEvaluationFactory(
                    outputRatio, minOutputs_, maxOutputs_);
                std::unique_ptr<IDecomposableRuleEvaluationFactory> pruningRuleEvaluationFactoryPtr =
                  labelBinningConfig_.get().createDecomposableFixedPartialRuleEvaluationFactory(
                    outputRatio, minOutputs_, maxOutputs_);
                return std::make_unique<DenseDecomposableRegressionStatisticsProviderFactory<StatisticType>>(
                  std::move(lossFactoryPtr), std::move(evaluationMeasureFactoryPtr),
                  std::move(defaultRuleEvaluationFactoryPtr), std::move(regularRuleEvaluationFactoryPtr),
                  std::move(pruningRuleEvaluationFactoryPtr), multiThreadingSettings);
            }

            std::unique_ptr<IRegressionStatisticsProviderFactory> createRegressionStatisticsProviderFactory(
              const IFeatureMatrix& featureMatrix, const IRowWiseRegressionMatrix& regressionMatrix,
              std::unique_ptr<INonDecomposableRegressionLossFactory<StatisticType>>& lossFactoryPtr,
              std::unique_ptr<IRegressionEvaluationMeasureFactory<StatisticType>>& evaluationMeasureFactoryPtr,
              const BlasFactory& blasFactory, const LapackFactory& lapackFactory) const override {
                MultiThreadingSettings multiThreadingSettings =
                  multiThreadingConfig_.get().getSettings(featureMatrix, regressionMatrix.getNumOutputs());
                float32 outputRatio = calculateOutputRatio(outputRatio_, regressionMatrix);
                std::unique_ptr<INonDecomposableRuleEvaluationFactory> defaultRuleEvaluationFactoryPtr =
                  labelBinningConfig_.get().createNonDecomposableCompleteRuleEvaluationFactory(blasFactory,
                                                                                               lapackFactory);
                std::unique_ptr<INonDecomposableRuleEvaluationFactory> regularRuleEvaluationFactoryPtr =
                  labelBinningConfig_.get().createNonDecomposableFixedPartialRuleEvaluationFactory(
                    outputRatio, minOutputs_, maxOutputs_, blasFactory, lapackFactory);
                std::unique_ptr<INonDecomposableRuleEvaluationFactory> pruningRuleEvaluationFactoryPtr =
                  labelBinningConfig_.get().createNonDecomposableFixedPartialRuleEvaluationFactory(
                    outputRatio, minOutputs_, maxOutputs_, blasFactory, lapackFactory);
                return std::make_unique<DenseNonDecomposableRegressionStatisticsProviderFactory<StatisticType>>(
                  std::move(lossFactoryPtr), std::move(evaluationMeasureFactoryPtr),
                  std::move(defaultRuleEvaluationFactoryPtr), std::move(regularRuleEvaluationFactoryPtr),
                  std::move(pruningRuleEvaluationFactoryPtr), multiThreadingSettings);
            }
    };

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

    std::unique_ptr<IHeadConfig::IPreset<float32>> FixedPartialHeadConfig::create32BitPreset() const {
        return std::make_unique<PartialFixedHeadPreset<float32>>(labelBinningConfig_, multiThreadingConfig_,
                                                                 outputRatio_, minOutputs_, maxOutputs_);
    }

    std::unique_ptr<IHeadConfig::IPreset<float64>> FixedPartialHeadConfig::create64BitPreset() const {
        return std::make_unique<PartialFixedHeadPreset<float64>>(labelBinningConfig_, multiThreadingConfig_,
                                                                 outputRatio_, minOutputs_, maxOutputs_);
    }

    bool FixedPartialHeadConfig::isPartial() const {
        return true;
    }

    bool FixedPartialHeadConfig::isSingleOutput() const {
        return false;
    }

}
