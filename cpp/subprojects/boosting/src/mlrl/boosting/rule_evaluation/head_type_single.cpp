#include "mlrl/boosting/rule_evaluation/head_type_single.hpp"

#include "mlrl/boosting/rule_evaluation/rule_evaluation_decomposable_single.hpp"
#include "mlrl/boosting/statistics/statistics_provider_decomposable_dense.hpp"
#include "mlrl/boosting/statistics/statistics_provider_decomposable_sparse.hpp"
#include "mlrl/boosting/statistics/statistics_provider_non_decomposable_dense.hpp"

namespace boosting {

    template<typename StatisticType>
    class SingleHeadPreset final : public IHeadConfig::IPreset<StatisticType> {
        private:

            ReadableProperty<ILabelBinningConfig> labelBinningConfig_;

            ReadableProperty<IMultiThreadingConfig> multiThreadingConfig_;

            ReadableProperty<IRegularizationConfig> l1RegularizationConfig_;

            ReadableProperty<IRegularizationConfig> l2RegularizationConfig_;

        public:

            SingleHeadPreset(ReadableProperty<ILabelBinningConfig> labelBinningConfig,
                             ReadableProperty<IMultiThreadingConfig> multiThreadingConfig,
                             ReadableProperty<IRegularizationConfig> l1RegularizationConfig,
                             ReadableProperty<IRegularizationConfig> l2RegularizationConfig)
                : labelBinningConfig_(labelBinningConfig), multiThreadingConfig_(multiThreadingConfig),
                  l1RegularizationConfig_(l1RegularizationConfig), l2RegularizationConfig_(l2RegularizationConfig) {}

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
                  std::make_unique<DecomposableSingleOutputRuleEvaluationFactory>(l1RegularizationWeight,
                                                                                  l2RegularizationWeight);
                std::unique_ptr<IDecomposableRuleEvaluationFactory> pruningRuleEvaluationFactoryPtr =
                  std::make_unique<DecomposableSingleOutputRuleEvaluationFactory>(l1RegularizationWeight,
                                                                                  l2RegularizationWeight);
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
                float32 l1RegularizationWeight = l1RegularizationConfig_.get().getWeight();
                float32 l2RegularizationWeight = l2RegularizationConfig_.get().getWeight();
                MultiThreadingSettings multiThreadingSettings =
                  multiThreadingConfig_.get().getSettings(featureMatrix, labelMatrix.getNumOutputs());
                std::unique_ptr<ISparseDecomposableRuleEvaluationFactory> regularRuleEvaluationFactoryPtr =
                  std::make_unique<DecomposableSingleOutputRuleEvaluationFactory>(l1RegularizationWeight,
                                                                                  l2RegularizationWeight);
                std::unique_ptr<ISparseDecomposableRuleEvaluationFactory> pruningRuleEvaluationFactoryPtr =
                  std::make_unique<DecomposableSingleOutputRuleEvaluationFactory>(l1RegularizationWeight,
                                                                                  l2RegularizationWeight);
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
                float32 l1RegularizationWeight = l1RegularizationConfig_.get().getWeight();
                float32 l2RegularizationWeight = l2RegularizationConfig_.get().getWeight();
                MultiThreadingSettings multiThreadingSettings =
                  multiThreadingConfig_.get().getSettings(featureMatrix, labelMatrix.getNumOutputs());
                std::unique_ptr<INonDecomposableRuleEvaluationFactory> defaultRuleEvaluationFactoryPtr =
                  labelBinningConfig_.get().createNonDecomposableCompleteRuleEvaluationFactory(blasFactory,
                                                                                               lapackFactory);
                std::unique_ptr<IDecomposableRuleEvaluationFactory> regularRuleEvaluationFactoryPtr =
                  std::make_unique<DecomposableSingleOutputRuleEvaluationFactory>(l1RegularizationWeight,
                                                                                  l2RegularizationWeight);
                std::unique_ptr<IDecomposableRuleEvaluationFactory> pruningRuleEvaluationFactoryPtr =
                  std::make_unique<DecomposableSingleOutputRuleEvaluationFactory>(l1RegularizationWeight,
                                                                                  l2RegularizationWeight);
                return std::make_unique<
                  DenseConvertibleNonDecomposableClassificationStatisticsProviderFactory<StatisticType>>(
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
                  std::make_unique<DecomposableSingleOutputRuleEvaluationFactory>(l1RegularizationWeight,
                                                                                  l2RegularizationWeight);
                std::unique_ptr<IDecomposableRuleEvaluationFactory> pruningRuleEvaluationFactoryPtr =
                  std::make_unique<DecomposableSingleOutputRuleEvaluationFactory>(l1RegularizationWeight,
                                                                                  l2RegularizationWeight);
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
                float32 l1RegularizationWeight = l1RegularizationConfig_.get().getWeight();
                float32 l2RegularizationWeight = l2RegularizationConfig_.get().getWeight();
                MultiThreadingSettings multiThreadingSettings =
                  multiThreadingConfig_.get().getSettings(featureMatrix, regressionMatrix.getNumOutputs());
                std::unique_ptr<INonDecomposableRuleEvaluationFactory> defaultRuleEvaluationFactoryPtr =
                  labelBinningConfig_.get().createNonDecomposableCompleteRuleEvaluationFactory(blasFactory,
                                                                                               lapackFactory);
                std::unique_ptr<IDecomposableRuleEvaluationFactory> regularRuleEvaluationFactoryPtr =
                  std::make_unique<DecomposableSingleOutputRuleEvaluationFactory>(l1RegularizationWeight,
                                                                                  l2RegularizationWeight);
                std::unique_ptr<IDecomposableRuleEvaluationFactory> pruningRuleEvaluationFactoryPtr =
                  std::make_unique<DecomposableSingleOutputRuleEvaluationFactory>(l1RegularizationWeight,
                                                                                  l2RegularizationWeight);
                return std::make_unique<
                  DenseConvertibleNonDecomposableRegressionStatisticsProviderFactory<StatisticType>>(
                  std::move(lossFactoryPtr), std::move(evaluationMeasureFactoryPtr),
                  std::move(defaultRuleEvaluationFactoryPtr), std::move(regularRuleEvaluationFactoryPtr),
                  std::move(pruningRuleEvaluationFactoryPtr), multiThreadingSettings);
            }
    };

    SingleOutputHeadConfig::SingleOutputHeadConfig(ReadableProperty<ILabelBinningConfig> labelBinningConfig,
                                                   ReadableProperty<IMultiThreadingConfig> multiThreadingConfig,
                                                   ReadableProperty<IRegularizationConfig> l1RegularizationConfig,
                                                   ReadableProperty<IRegularizationConfig> l2RegularizationConfig)
        : labelBinningConfig_(labelBinningConfig), multiThreadingConfig_(multiThreadingConfig),
          l1RegularizationConfig_(l1RegularizationConfig), l2RegularizationConfig_(l2RegularizationConfig) {}

    std::unique_ptr<IHeadConfig::IPreset<float32>> SingleOutputHeadConfig::create32BitPreset() const {
        return std::make_unique<SingleHeadPreset<float32>>(labelBinningConfig_, multiThreadingConfig_,
                                                           l1RegularizationConfig_, l2RegularizationConfig_);
    }

    std::unique_ptr<IHeadConfig::IPreset<float64>> SingleOutputHeadConfig::create64BitPreset() const {
        return std::make_unique<SingleHeadPreset<float64>>(labelBinningConfig_, multiThreadingConfig_,
                                                           l1RegularizationConfig_, l2RegularizationConfig_);
    }

    bool SingleOutputHeadConfig::isPartial() const {
        return true;
    }

    bool SingleOutputHeadConfig::isSingleOutput() const {
        return true;
    }

}
