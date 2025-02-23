#include "mlrl/boosting/rule_evaluation/head_type_complete.hpp"

#include "mlrl/boosting/rule_evaluation/rule_evaluation_decomposable_complete.hpp"
#include "mlrl/boosting/statistics/statistics_provider_decomposable_dense.hpp"
#include "mlrl/boosting/statistics/statistics_provider_non_decomposable_dense.hpp"

namespace boosting {

    CompleteHeadConfig::CompleteHeadConfig(ReadableProperty<ILabelBinningConfig> labelBinningConfig,
                                           ReadableProperty<IMultiThreadingConfig> multiThreadingConfig,
                                           ReadableProperty<IRegularizationConfig> l1RegularizationConfig,
                                           ReadableProperty<IRegularizationConfig> l2RegularizationConfig)
        : labelBinningConfig_(labelBinningConfig), multiThreadingConfig_(multiThreadingConfig),
          l1RegularizationConfig_(l1RegularizationConfig), l2RegularizationConfig_(l2RegularizationConfig) {}

    std::unique_ptr<IClassificationStatisticsProviderFactory>
      CompleteHeadConfig::createClassificationStatisticsProviderFactory(
        const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix,
        std::unique_ptr<IDecomposableClassificationLossFactory<float64>>& lossFactoryPtr,
        std::unique_ptr<IClassificationEvaluationMeasureFactory<float64>>& evaluationMeasureFactoryPtr) const {
        float32 l1RegularizationWeight = l1RegularizationConfig_.get().getWeight();
        float32 l2RegularizationWeight = l2RegularizationConfig_.get().getWeight();
        MultiThreadingSettings multiThreadingSettings =
          multiThreadingConfig_.get().getSettings(featureMatrix, labelMatrix.getNumOutputs());
        std::unique_ptr<IDecomposableRuleEvaluationFactory> defaultRuleEvaluationFactoryPtr =
          labelBinningConfig_.get().createDecomposableCompleteRuleEvaluationFactory();
        std::unique_ptr<IDecomposableRuleEvaluationFactory> regularRuleEvaluationFactoryPtr =
          std::make_unique<DecomposableCompleteRuleEvaluationFactory>(l1RegularizationWeight, l2RegularizationWeight);
        std::unique_ptr<IDecomposableRuleEvaluationFactory> pruningRuleEvaluationFactoryPtr =
          std::make_unique<DecomposableCompleteRuleEvaluationFactory>(l1RegularizationWeight, l2RegularizationWeight);
        return std::make_unique<DenseDecomposableClassificationStatisticsProviderFactory<float64>>(
          std::move(lossFactoryPtr), std::move(evaluationMeasureFactoryPtr), std::move(defaultRuleEvaluationFactoryPtr),
          std::move(regularRuleEvaluationFactoryPtr), std::move(pruningRuleEvaluationFactoryPtr),
          multiThreadingSettings);
    }

    std::unique_ptr<IClassificationStatisticsProviderFactory>
      CompleteHeadConfig::createClassificationStatisticsProviderFactory(
        const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix,
        std::unique_ptr<ISparseDecomposableClassificationLossFactory<float64>>& lossFactoryPtr,
        std::unique_ptr<ISparseEvaluationMeasureFactory<float64>>& evaluationMeasureFactoryPtr) const {
        float32 l1RegularizationWeight = l1RegularizationConfig_.get().getWeight();
        float32 l2RegularizationWeight = l2RegularizationConfig_.get().getWeight();
        MultiThreadingSettings multiThreadingSettings =
          multiThreadingConfig_.get().getSettings(featureMatrix, labelMatrix.getNumOutputs());
        std::unique_ptr<IDecomposableRuleEvaluationFactory> defaultRuleEvaluationFactoryPtr =
          labelBinningConfig_.get().createDecomposableCompleteRuleEvaluationFactory();
        std::unique_ptr<IDecomposableRuleEvaluationFactory> regularRuleEvaluationFactoryPtr =
          std::make_unique<DecomposableCompleteRuleEvaluationFactory>(l1RegularizationWeight, l2RegularizationWeight);
        std::unique_ptr<IDecomposableRuleEvaluationFactory> pruningRuleEvaluationFactoryPtr =
          std::make_unique<DecomposableCompleteRuleEvaluationFactory>(l1RegularizationWeight, l2RegularizationWeight);
        return std::make_unique<DenseDecomposableClassificationStatisticsProviderFactory<float64>>(
          std::move(lossFactoryPtr), std::move(evaluationMeasureFactoryPtr), std::move(defaultRuleEvaluationFactoryPtr),
          std::move(regularRuleEvaluationFactoryPtr), std::move(pruningRuleEvaluationFactoryPtr),
          multiThreadingSettings);
    }

    std::unique_ptr<IClassificationStatisticsProviderFactory>
      CompleteHeadConfig::createClassificationStatisticsProviderFactory(
        const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix,
        std::unique_ptr<INonDecomposableClassificationLossFactory<float64>>& lossFactoryPtr,
        std::unique_ptr<IClassificationEvaluationMeasureFactory<float64>>& evaluationMeasureFactoryPtr,
        const BlasFactory& blasFactory, const LapackFactory& lapackFactory) const {
        MultiThreadingSettings multiThreadingSettings =
          multiThreadingConfig_.get().getSettings(featureMatrix, labelMatrix.getNumOutputs());
        std::unique_ptr<INonDecomposableRuleEvaluationFactory> defaultRuleEvaluationFactoryPtr =
          labelBinningConfig_.get().createNonDecomposableCompleteRuleEvaluationFactory(blasFactory, lapackFactory);
        std::unique_ptr<INonDecomposableRuleEvaluationFactory> regularRuleEvaluationFactoryPtr =
          labelBinningConfig_.get().createNonDecomposableCompleteRuleEvaluationFactory(blasFactory, lapackFactory);
        std::unique_ptr<INonDecomposableRuleEvaluationFactory> pruningRuleEvaluationFactoryPtr =
          labelBinningConfig_.get().createNonDecomposableCompleteRuleEvaluationFactory(blasFactory, lapackFactory);
        return std::make_unique<DenseNonDecomposableClassificationStatisticsProviderFactory<float64>>(
          std::move(lossFactoryPtr), std::move(evaluationMeasureFactoryPtr), std::move(defaultRuleEvaluationFactoryPtr),
          std::move(regularRuleEvaluationFactoryPtr), std::move(pruningRuleEvaluationFactoryPtr),
          multiThreadingSettings);
    }

    std::unique_ptr<IRegressionStatisticsProviderFactory> CompleteHeadConfig::createRegressionStatisticsProviderFactory(
      const IFeatureMatrix& featureMatrix, const IRowWiseRegressionMatrix& regressionMatrix,
      std::unique_ptr<IDecomposableRegressionLossFactory<float64>>& lossFactoryPtr,
      std::unique_ptr<IRegressionEvaluationMeasureFactory<float64>>& evaluationMeasureFactoryPtr) const {
        float32 l1RegularizationWeight = l1RegularizationConfig_.get().getWeight();
        float32 l2RegularizationWeight = l2RegularizationConfig_.get().getWeight();
        MultiThreadingSettings multiThreadingSettings =
          multiThreadingConfig_.get().getSettings(featureMatrix, regressionMatrix.getNumOutputs());
        std::unique_ptr<IDecomposableRuleEvaluationFactory> defaultRuleEvaluationFactoryPtr =
          labelBinningConfig_.get().createDecomposableCompleteRuleEvaluationFactory();
        std::unique_ptr<IDecomposableRuleEvaluationFactory> regularRuleEvaluationFactoryPtr =
          std::make_unique<DecomposableCompleteRuleEvaluationFactory>(l1RegularizationWeight, l2RegularizationWeight);
        std::unique_ptr<IDecomposableRuleEvaluationFactory> pruningRuleEvaluationFactoryPtr =
          std::make_unique<DecomposableCompleteRuleEvaluationFactory>(l1RegularizationWeight, l2RegularizationWeight);
        return std::make_unique<DenseDecomposableRegressionStatisticsProviderFactory<float64>>(
          std::move(lossFactoryPtr), std::move(evaluationMeasureFactoryPtr), std::move(defaultRuleEvaluationFactoryPtr),
          std::move(regularRuleEvaluationFactoryPtr), std::move(pruningRuleEvaluationFactoryPtr),
          multiThreadingSettings);
    }

    std::unique_ptr<IRegressionStatisticsProviderFactory> CompleteHeadConfig::createRegressionStatisticsProviderFactory(
      const IFeatureMatrix& featureMatrix, const IRowWiseRegressionMatrix& regressionMatrix,
      std::unique_ptr<INonDecomposableRegressionLossFactory<float64>>& lossFactoryPtr,
      std::unique_ptr<IRegressionEvaluationMeasureFactory<float64>>& evaluationMeasureFactoryPtr,
      const BlasFactory& blasFactory, const LapackFactory& lapackFactory) const {
        MultiThreadingSettings multiThreadingSettings =
          multiThreadingConfig_.get().getSettings(featureMatrix, regressionMatrix.getNumOutputs());
        std::unique_ptr<INonDecomposableRuleEvaluationFactory> defaultRuleEvaluationFactoryPtr =
          labelBinningConfig_.get().createNonDecomposableCompleteRuleEvaluationFactory(blasFactory, lapackFactory);
        std::unique_ptr<INonDecomposableRuleEvaluationFactory> regularRuleEvaluationFactoryPtr =
          labelBinningConfig_.get().createNonDecomposableCompleteRuleEvaluationFactory(blasFactory, lapackFactory);
        std::unique_ptr<INonDecomposableRuleEvaluationFactory> pruningRuleEvaluationFactoryPtr =
          labelBinningConfig_.get().createNonDecomposableCompleteRuleEvaluationFactory(blasFactory, lapackFactory);
        return std::make_unique<DenseNonDecomposableRegressionStatisticsProviderFactory<float64>>(
          std::move(lossFactoryPtr), std::move(evaluationMeasureFactoryPtr), std::move(defaultRuleEvaluationFactoryPtr),
          std::move(regularRuleEvaluationFactoryPtr), std::move(pruningRuleEvaluationFactoryPtr),
          multiThreadingSettings);
    }

    bool CompleteHeadConfig::isPartial() const {
        return false;
    }

    bool CompleteHeadConfig::isSingleOutput() const {
        return false;
    }

}
