#include "mlrl/boosting/rule_evaluation/head_type_complete.hpp"

#include "mlrl/boosting/rule_evaluation/rule_evaluation_decomposable_complete.hpp"
#include "mlrl/boosting/statistics/statistics_provider_decomposable_dense.hpp"
#include "mlrl/boosting/statistics/statistics_provider_non_decomposable_dense.hpp"
#include "mlrl/common/math/vector_math.hpp"
#include "mlrl/common/util/xsimd.hpp"

namespace boosting {

    template<typename StatisticType>
    class CompleteHeadPreset final : public IHeadConfig::IPreset<StatisticType> {
        private:

            ReadableProperty<ILabelBinningConfig> labelBinningConfig_;

            ReadableProperty<IMultiThreadingConfig> multiThreadingConfig_;

            ReadableProperty<ISimdConfig> simdConfig_;

            ReadableProperty<IRegularizationConfig> l1RegularizationConfig_;

            ReadableProperty<IRegularizationConfig> l2RegularizationConfig_;

        public:

            CompleteHeadPreset(ReadableProperty<ILabelBinningConfig> labelBinningConfig,
                               ReadableProperty<IMultiThreadingConfig> multiThreadingConfig,
                               ReadableProperty<ISimdConfig> simdConfig,
                               ReadableProperty<IRegularizationConfig> l1RegularizationConfig,
                               ReadableProperty<IRegularizationConfig> l2RegularizationConfig)
                : labelBinningConfig_(labelBinningConfig), multiThreadingConfig_(multiThreadingConfig),
                  simdConfig_(simdConfig), l1RegularizationConfig_(l1RegularizationConfig),
                  l2RegularizationConfig_(l2RegularizationConfig) {}

            std::unique_ptr<IClassificationStatisticsProviderFactory> createClassificationStatisticsProviderFactory(
              const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix,
              std::unique_ptr<IDecomposableClassificationLossFactory<StatisticType>>& lossFactoryPtr,
              std::unique_ptr<IClassificationEvaluationMeasureFactory<StatisticType>>& evaluationMeasureFactoryPtr)
              const override {
                float32 l1RegularizationWeight = l1RegularizationConfig_.get().getWeight();
                float32 l2RegularizationWeight = l2RegularizationConfig_.get().getWeight();
                MultiThreadingSettings multiThreadingSettings =
                  multiThreadingConfig_.get().getSettings(featureMatrix, labelMatrix.getNumOutputs());
                std::unique_ptr<IDecomposableRuleEvaluationFactory> defaultRuleEvaluationFactoryPtr =
                  labelBinningConfig_.get().createDecomposableCompleteRuleEvaluationFactory();
                std::unique_ptr<IDecomposableRuleEvaluationFactory> regularRuleEvaluationFactoryPtr =
                  std::make_unique<DecomposableCompleteRuleEvaluationFactory>(l1RegularizationWeight,
                                                                              l2RegularizationWeight);
                std::unique_ptr<IDecomposableRuleEvaluationFactory> pruningRuleEvaluationFactoryPtr =
                  std::make_unique<DecomposableCompleteRuleEvaluationFactory>(l1RegularizationWeight,
                                                                              l2RegularizationWeight);

#if SIMD_SUPPORT_ENABLED
                if (labelMatrix.getNumOutputs() > 1 && simdConfig_.get().isSimdEnabled()) {
                    return std::make_unique<
                      DenseDecomposableClassificationStatisticsProviderFactory<StatisticType, SimdVectorMath>>(
                      std::move(lossFactoryPtr), std::move(evaluationMeasureFactoryPtr),
                      std::move(defaultRuleEvaluationFactoryPtr), std::move(regularRuleEvaluationFactoryPtr),
                      std::move(pruningRuleEvaluationFactoryPtr), multiThreadingSettings);
                }
#endif

                return std::make_unique<
                  DenseDecomposableClassificationStatisticsProviderFactory<StatisticType, SequentialVectorMath>>(
                  std::move(lossFactoryPtr), std::move(evaluationMeasureFactoryPtr),
                  std::move(defaultRuleEvaluationFactoryPtr), std::move(regularRuleEvaluationFactoryPtr),
                  std::move(pruningRuleEvaluationFactoryPtr), multiThreadingSettings);
            }

            std::unique_ptr<IClassificationStatisticsProviderFactory> createClassificationStatisticsProviderFactory(
              const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix,
              std::unique_ptr<ISparseDecomposableClassificationLossFactory<StatisticType>>& lossFactoryPtr,
              std::unique_ptr<ISparseEvaluationMeasureFactory<StatisticType>>& evaluationMeasureFactoryPtr)
              const override {
                float32 l1RegularizationWeight = l1RegularizationConfig_.get().getWeight();
                float32 l2RegularizationWeight = l2RegularizationConfig_.get().getWeight();
                MultiThreadingSettings multiThreadingSettings =
                  multiThreadingConfig_.get().getSettings(featureMatrix, labelMatrix.getNumOutputs());
                std::unique_ptr<IDecomposableRuleEvaluationFactory> defaultRuleEvaluationFactoryPtr =
                  labelBinningConfig_.get().createDecomposableCompleteRuleEvaluationFactory();
                std::unique_ptr<IDecomposableRuleEvaluationFactory> regularRuleEvaluationFactoryPtr =
                  std::make_unique<DecomposableCompleteRuleEvaluationFactory>(l1RegularizationWeight,
                                                                              l2RegularizationWeight);
                std::unique_ptr<IDecomposableRuleEvaluationFactory> pruningRuleEvaluationFactoryPtr =
                  std::make_unique<DecomposableCompleteRuleEvaluationFactory>(l1RegularizationWeight,
                                                                              l2RegularizationWeight);

#if SIMD_SUPPORT_ENABLED
                if (labelMatrix.getNumOutputs() > 1 && simdConfig_.get().isSimdEnabled()) {
                    return std::make_unique<
                      DenseDecomposableClassificationStatisticsProviderFactory<StatisticType, SimdVectorMath>>(
                      std::move(lossFactoryPtr), std::move(evaluationMeasureFactoryPtr),
                      std::move(defaultRuleEvaluationFactoryPtr), std::move(regularRuleEvaluationFactoryPtr),
                      std::move(pruningRuleEvaluationFactoryPtr), multiThreadingSettings);
                }
#endif

                return std::make_unique<
                  DenseDecomposableClassificationStatisticsProviderFactory<StatisticType, SequentialVectorMath>>(
                  std::move(lossFactoryPtr), std::move(evaluationMeasureFactoryPtr),
                  std::move(defaultRuleEvaluationFactoryPtr), std::move(regularRuleEvaluationFactoryPtr),
                  std::move(pruningRuleEvaluationFactoryPtr), multiThreadingSettings);
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
                  labelBinningConfig_.get().createNonDecomposableCompleteRuleEvaluationFactory(blasFactory,
                                                                                               lapackFactory);
                std::unique_ptr<INonDecomposableRuleEvaluationFactory> pruningRuleEvaluationFactoryPtr =
                  labelBinningConfig_.get().createNonDecomposableCompleteRuleEvaluationFactory(blasFactory,
                                                                                               lapackFactory);

#if SIMD_SUPPORT_ENABLED
                if (labelMatrix.getNumOutputs() > 1 && simdConfig_.get().isSimdEnabled()) {
                    return std::make_unique<
                      DenseNonDecomposableClassificationStatisticsProviderFactory<StatisticType, SimdVectorMath>>(
                      std::move(lossFactoryPtr), std::move(evaluationMeasureFactoryPtr),
                      std::move(defaultRuleEvaluationFactoryPtr), std::move(regularRuleEvaluationFactoryPtr),
                      std::move(pruningRuleEvaluationFactoryPtr), multiThreadingSettings);
                }
#endif

                return std::make_unique<
                  DenseNonDecomposableClassificationStatisticsProviderFactory<StatisticType, SequentialVectorMath>>(
                  std::move(lossFactoryPtr), std::move(evaluationMeasureFactoryPtr),
                  std::move(defaultRuleEvaluationFactoryPtr), std::move(regularRuleEvaluationFactoryPtr),
                  std::move(pruningRuleEvaluationFactoryPtr), multiThreadingSettings);
            }

            std::unique_ptr<IRegressionStatisticsProviderFactory> createRegressionStatisticsProviderFactory(
              const IFeatureMatrix& featureMatrix, const IRowWiseRegressionMatrix& regressionMatrix,
              std::unique_ptr<IDecomposableRegressionLossFactory<StatisticType>>& lossFactoryPtr,
              std::unique_ptr<IRegressionEvaluationMeasureFactory<StatisticType>>& evaluationMeasureFactoryPtr)
              const override {
                float32 l1RegularizationWeight = l1RegularizationConfig_.get().getWeight();
                float32 l2RegularizationWeight = l2RegularizationConfig_.get().getWeight();
                MultiThreadingSettings multiThreadingSettings =
                  multiThreadingConfig_.get().getSettings(featureMatrix, regressionMatrix.getNumOutputs());
                std::unique_ptr<IDecomposableRuleEvaluationFactory> defaultRuleEvaluationFactoryPtr =
                  labelBinningConfig_.get().createDecomposableCompleteRuleEvaluationFactory();
                std::unique_ptr<IDecomposableRuleEvaluationFactory> regularRuleEvaluationFactoryPtr =
                  std::make_unique<DecomposableCompleteRuleEvaluationFactory>(l1RegularizationWeight,
                                                                              l2RegularizationWeight);
                std::unique_ptr<IDecomposableRuleEvaluationFactory> pruningRuleEvaluationFactoryPtr =
                  std::make_unique<DecomposableCompleteRuleEvaluationFactory>(l1RegularizationWeight,
                                                                              l2RegularizationWeight);

#if SIMD_SUPPORT_ENABLED
                if (regressionMatrix.getNumOutputs() > 1 && simdConfig_.get().isSimdEnabled()) {
                    return std::make_unique<
                      DenseDecomposableRegressionStatisticsProviderFactory<StatisticType, SimdVectorMath>>(
                      std::move(lossFactoryPtr), std::move(evaluationMeasureFactoryPtr),
                      std::move(defaultRuleEvaluationFactoryPtr), std::move(regularRuleEvaluationFactoryPtr),
                      std::move(pruningRuleEvaluationFactoryPtr), multiThreadingSettings);
                }
#endif

                return std::make_unique<
                  DenseDecomposableRegressionStatisticsProviderFactory<StatisticType, SequentialVectorMath>>(
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
                  labelBinningConfig_.get().createNonDecomposableCompleteRuleEvaluationFactory(blasFactory,
                                                                                               lapackFactory);
                std::unique_ptr<INonDecomposableRuleEvaluationFactory> pruningRuleEvaluationFactoryPtr =
                  labelBinningConfig_.get().createNonDecomposableCompleteRuleEvaluationFactory(blasFactory,
                                                                                               lapackFactory);

#if SIMD_SUPPORT_ENABLED
                if (regressionMatrix.getNumOutputs() > 1 && simdConfig_.get().isSimdEnabled()) {
                    return std::make_unique<
                      DenseNonDecomposableRegressionStatisticsProviderFactory<StatisticType, SimdVectorMath>>(
                      std::move(lossFactoryPtr), std::move(evaluationMeasureFactoryPtr),
                      std::move(defaultRuleEvaluationFactoryPtr), std::move(regularRuleEvaluationFactoryPtr),
                      std::move(pruningRuleEvaluationFactoryPtr), multiThreadingSettings);
                }
#endif

                return std::make_unique<
                  DenseNonDecomposableRegressionStatisticsProviderFactory<StatisticType, SequentialVectorMath>>(
                  std::move(lossFactoryPtr), std::move(evaluationMeasureFactoryPtr),
                  std::move(defaultRuleEvaluationFactoryPtr), std::move(regularRuleEvaluationFactoryPtr),
                  std::move(pruningRuleEvaluationFactoryPtr), multiThreadingSettings);
            }
    };

    CompleteHeadConfig::CompleteHeadConfig(ReadableProperty<ILabelBinningConfig> labelBinningConfig,
                                           ReadableProperty<IMultiThreadingConfig> multiThreadingConfig,
                                           ReadableProperty<ISimdConfig> simdConfig,
                                           ReadableProperty<IRegularizationConfig> l1RegularizationConfig,
                                           ReadableProperty<IRegularizationConfig> l2RegularizationConfig)
        : labelBinningConfig_(labelBinningConfig), multiThreadingConfig_(multiThreadingConfig), simdConfig_(simdConfig),
          l1RegularizationConfig_(l1RegularizationConfig), l2RegularizationConfig_(l2RegularizationConfig) {}

    std::unique_ptr<IHeadConfig::IPreset<float32>> CompleteHeadConfig::create32BitPreset() const {
        return std::make_unique<CompleteHeadPreset<float32>>(labelBinningConfig_, multiThreadingConfig_, simdConfig_,
                                                             l1RegularizationConfig_, l2RegularizationConfig_);
    }

    std::unique_ptr<IHeadConfig::IPreset<float64>> CompleteHeadConfig::create64BitPreset() const {
        return std::make_unique<CompleteHeadPreset<float64>>(labelBinningConfig_, multiThreadingConfig_, simdConfig_,
                                                             l1RegularizationConfig_, l2RegularizationConfig_);
    }

    bool CompleteHeadConfig::isPartial() const {
        return false;
    }

    bool CompleteHeadConfig::isSingleOutput() const {
        return false;
    }

}
