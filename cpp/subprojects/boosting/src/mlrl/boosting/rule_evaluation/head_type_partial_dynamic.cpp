#include "mlrl/boosting/rule_evaluation/head_type_partial_dynamic.hpp"

#include "mlrl/boosting/statistics/statistics_provider_decomposable_dense.hpp"
#include "mlrl/boosting/statistics/statistics_provider_decomposable_sparse.hpp"
#include "mlrl/boosting/statistics/statistics_provider_non_decomposable_dense.hpp"
#include "mlrl/common/util/validation.hpp"

namespace boosting {

    template<typename StatisticType>
    class PartialDynamicHeadPreset final : public IHeadConfig::IPreset<StatisticType> {
        private:

            ReadableProperty<ILabelBinningConfig> labelBinningConfig_;

            ReadableProperty<IMultiThreadingConfig> multiThreadingConfig_;

            float32 threshold_;

            float32 exponent_;

        public:

            PartialDynamicHeadPreset(ReadableProperty<ILabelBinningConfig> labelBinningConfig,
                                     ReadableProperty<IMultiThreadingConfig> multiThreadingConfig, float32 threshold,
                                     float32 exponent)
                : labelBinningConfig_(labelBinningConfig), multiThreadingConfig_(multiThreadingConfig),
                  threshold_(threshold), exponent_(exponent) {}

            std::unique_ptr<IClassificationStatisticsProviderFactory> createClassificationStatisticsProviderFactory(
              const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix,
              std::unique_ptr<IDecomposableClassificationLossFactory<StatisticType>>& lossFactoryPtr,
              std::unique_ptr<IClassificationEvaluationMeasureFactory<StatisticType>>& evaluationMeasureFactoryPtr)
              const override {
                MultiThreadingSettings multiThreadingSettings =
                  multiThreadingConfig_.get().getSettings(featureMatrix, labelMatrix.getNumOutputs());
                std::unique_ptr<IDecomposableRuleEvaluationFactory> defaultRuleEvaluationFactoryPtr =
                  labelBinningConfig_.get().createDecomposableCompleteRuleEvaluationFactory();
                std::unique_ptr<IDecomposableRuleEvaluationFactory> regularRuleEvaluationFactoryPtr =
                  labelBinningConfig_.get().createDecomposableDynamicPartialRuleEvaluationFactory(threshold_,
                                                                                                  exponent_);
                std::unique_ptr<IDecomposableRuleEvaluationFactory> pruningRuleEvaluationFactoryPtr =
                  labelBinningConfig_.get().createDecomposableDynamicPartialRuleEvaluationFactory(threshold_,
                                                                                                  exponent_);
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
                std::unique_ptr<ISparseDecomposableRuleEvaluationFactory> regularRuleEvaluationFactoryPtr =
                  labelBinningConfig_.get().createDecomposableDynamicPartialRuleEvaluationFactory(threshold_,
                                                                                                  exponent_);
                std::unique_ptr<ISparseDecomposableRuleEvaluationFactory> pruningRuleEvaluationFactoryPtr =
                  labelBinningConfig_.get().createDecomposableDynamicPartialRuleEvaluationFactory(threshold_,
                                                                                                  exponent_);
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
                std::unique_ptr<INonDecomposableRuleEvaluationFactory> defaultRuleEvaluationFactoryPtr =
                  labelBinningConfig_.get().createNonDecomposableCompleteRuleEvaluationFactory(blasFactory,
                                                                                               lapackFactory);
                std::unique_ptr<INonDecomposableRuleEvaluationFactory> regularRuleEvaluationFactoryPtr =
                  labelBinningConfig_.get().createNonDecomposableDynamicPartialRuleEvaluationFactory(
                    threshold_, exponent_, blasFactory, lapackFactory);
                std::unique_ptr<INonDecomposableRuleEvaluationFactory> pruningRuleEvaluationFactoryPtr =
                  labelBinningConfig_.get().createNonDecomposableDynamicPartialRuleEvaluationFactory(
                    threshold_, exponent_, blasFactory, lapackFactory);
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
                std::unique_ptr<IDecomposableRuleEvaluationFactory> defaultRuleEvaluationFactoryPtr =
                  labelBinningConfig_.get().createDecomposableCompleteRuleEvaluationFactory();
                std::unique_ptr<IDecomposableRuleEvaluationFactory> regularRuleEvaluationFactoryPtr =
                  labelBinningConfig_.get().createDecomposableDynamicPartialRuleEvaluationFactory(threshold_,
                                                                                                  exponent_);
                std::unique_ptr<IDecomposableRuleEvaluationFactory> pruningRuleEvaluationFactoryPtr =
                  labelBinningConfig_.get().createDecomposableDynamicPartialRuleEvaluationFactory(threshold_,
                                                                                                  exponent_);
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
                std::unique_ptr<INonDecomposableRuleEvaluationFactory> defaultRuleEvaluationFactoryPtr =
                  labelBinningConfig_.get().createNonDecomposableCompleteRuleEvaluationFactory(blasFactory,
                                                                                               lapackFactory);
                std::unique_ptr<INonDecomposableRuleEvaluationFactory> regularRuleEvaluationFactoryPtr =
                  labelBinningConfig_.get().createNonDecomposableDynamicPartialRuleEvaluationFactory(
                    threshold_, exponent_, blasFactory, lapackFactory);
                std::unique_ptr<INonDecomposableRuleEvaluationFactory> pruningRuleEvaluationFactoryPtr =
                  labelBinningConfig_.get().createNonDecomposableDynamicPartialRuleEvaluationFactory(
                    threshold_, exponent_, blasFactory, lapackFactory);
                return std::make_unique<DenseNonDecomposableRegressionStatisticsProviderFactory<StatisticType>>(
                  std::move(lossFactoryPtr), std::move(evaluationMeasureFactoryPtr),
                  std::move(defaultRuleEvaluationFactoryPtr), std::move(regularRuleEvaluationFactoryPtr),
                  std::move(pruningRuleEvaluationFactoryPtr), multiThreadingSettings);
            }
    };

    DynamicPartialHeadConfig::DynamicPartialHeadConfig(ReadableProperty<ILabelBinningConfig> labelBinningConfig,
                                                       ReadableProperty<IMultiThreadingConfig> multiThreadingConfig)
        : threshold_(0.02f), exponent_(2.0f), labelBinningConfig_(labelBinningConfig),
          multiThreadingConfig_(multiThreadingConfig) {}

    float32 DynamicPartialHeadConfig::getThreshold() const {
        return threshold_;
    }

    IDynamicPartialHeadConfig& DynamicPartialHeadConfig::setThreshold(float32 threshold) {
        util::assertGreater<float32>("threshold", threshold, 0);
        util::assertLess<float32>("threshold", threshold, 1);
        threshold_ = threshold;
        return *this;
    }

    float32 DynamicPartialHeadConfig::getExponent() const {
        return exponent_;
    }

    IDynamicPartialHeadConfig& DynamicPartialHeadConfig::setExponent(float32 exponent) {
        util::assertGreaterOrEqual<float32>("exponent", exponent, 1);
        exponent_ = exponent;
        return *this;
    }

    std::unique_ptr<IHeadConfig::IPreset<float32>> DynamicPartialHeadConfig::create32BitPreset() const {
        return std::make_unique<PartialDynamicHeadPreset<float32>>(labelBinningConfig_, multiThreadingConfig_,
                                                                   threshold_, exponent_);
    }

    std::unique_ptr<IHeadConfig::IPreset<float64>> DynamicPartialHeadConfig::create64BitPreset() const {
        return std::make_unique<PartialDynamicHeadPreset<float64>>(labelBinningConfig_, multiThreadingConfig_,
                                                                   threshold_, exponent_);
    }

    bool DynamicPartialHeadConfig::isPartial() const {
        return true;
    }

    bool DynamicPartialHeadConfig::isSingleOutput() const {
        return false;
    }

}
