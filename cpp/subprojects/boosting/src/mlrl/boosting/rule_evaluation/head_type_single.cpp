#include "mlrl/boosting/rule_evaluation/head_type_single.hpp"

#include "mlrl/boosting/rule_evaluation/rule_evaluation_decomposable_single.hpp"
#include "mlrl/boosting/statistics/statistics_provider_decomposable_dense.hpp"
#include "mlrl/boosting/statistics/statistics_provider_decomposable_sparse.hpp"
#include "mlrl/boosting/statistics/statistics_provider_non_decomposable_dense.hpp"

namespace boosting {

    SingleOutputHeadConfig::SingleOutputHeadConfig(ReadableProperty<ILabelBinningConfig> labelBinningConfig,
                                                   ReadableProperty<IMultiThreadingConfig> multiThreadingConfig,
                                                   ReadableProperty<IRegularizationConfig> l1RegularizationConfig,
                                                   ReadableProperty<IRegularizationConfig> l2RegularizationConfig)
        : labelBinningConfig_(labelBinningConfig), multiThreadingConfig_(multiThreadingConfig),
          l1RegularizationConfig_(l1RegularizationConfig), l2RegularizationConfig_(l2RegularizationConfig) {}

    std::unique_ptr<IClassificationStatisticsProviderFactory>
      SingleOutputHeadConfig::createClassificationStatisticsProviderFactory(
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
          std::make_unique<DecomposableSingleOutputRuleEvaluationFactory>(l1RegularizationWeight,
                                                                          l2RegularizationWeight);
        std::unique_ptr<IDecomposableRuleEvaluationFactory> pruningRuleEvaluationFactoryPtr =
          std::make_unique<DecomposableSingleOutputRuleEvaluationFactory>(l1RegularizationWeight,
                                                                          l2RegularizationWeight);
        return std::make_unique<DenseDecomposableClassificationStatisticsProviderFactory<float64>>(
          std::move(lossFactoryPtr), std::move(evaluationMeasureFactoryPtr), std::move(defaultRuleEvaluationFactoryPtr),
          std::move(regularRuleEvaluationFactoryPtr), std::move(pruningRuleEvaluationFactoryPtr),
          multiThreadingSettings);
    }

    std::unique_ptr<IClassificationStatisticsProviderFactory>
      SingleOutputHeadConfig::createClassificationStatisticsProviderFactory(
        const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix,
        std::unique_ptr<ISparseDecomposableClassificationLossFactory<float64>>& lossFactoryPtr,
        std::unique_ptr<ISparseEvaluationMeasureFactory<float64>>& evaluationMeasureFactoryPtr) const {
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
        return std::make_unique<SparseDecomposableClassificationStatisticsProviderFactory<float64>>(
          std::move(lossFactoryPtr), std::move(evaluationMeasureFactoryPtr), std::move(regularRuleEvaluationFactoryPtr),
          std::move(pruningRuleEvaluationFactoryPtr), multiThreadingSettings);
    }

    std::unique_ptr<IClassificationStatisticsProviderFactory>
      SingleOutputHeadConfig::createClassificationStatisticsProviderFactory(
        const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix,
        std::unique_ptr<INonDecomposableClassificationLossFactory<float64>>& lossFactoryPtr,
        std::unique_ptr<IClassificationEvaluationMeasureFactory<float64>>& evaluationMeasureFactoryPtr,
        const BlasFactory& blasFactory, const LapackFactory& lapackFactory) const {
        float32 l1RegularizationWeight = l1RegularizationConfig_.get().getWeight();
        float32 l2RegularizationWeight = l2RegularizationConfig_.get().getWeight();
        MultiThreadingSettings multiThreadingSettings =
          multiThreadingConfig_.get().getSettings(featureMatrix, labelMatrix.getNumOutputs());
        std::unique_ptr<INonDecomposableRuleEvaluationFactory> defaultRuleEvaluationFactoryPtr =
          labelBinningConfig_.get().createNonDecomposableCompleteRuleEvaluationFactory(blasFactory, lapackFactory);
        std::unique_ptr<IDecomposableRuleEvaluationFactory> regularRuleEvaluationFactoryPtr =
          std::make_unique<DecomposableSingleOutputRuleEvaluationFactory>(l1RegularizationWeight,
                                                                          l2RegularizationWeight);
        std::unique_ptr<IDecomposableRuleEvaluationFactory> pruningRuleEvaluationFactoryPtr =
          std::make_unique<DecomposableSingleOutputRuleEvaluationFactory>(l1RegularizationWeight,
                                                                          l2RegularizationWeight);
        return std::make_unique<DenseConvertibleNonDecomposableClassificationStatisticsProviderFactory<float64>>(
          std::move(lossFactoryPtr), std::move(evaluationMeasureFactoryPtr), std::move(defaultRuleEvaluationFactoryPtr),
          std::move(regularRuleEvaluationFactoryPtr), std::move(pruningRuleEvaluationFactoryPtr),
          multiThreadingSettings);
    }

    std::unique_ptr<IRegressionStatisticsProviderFactory>
      SingleOutputHeadConfig::createRegressionStatisticsProviderFactory(
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
          std::make_unique<DecomposableSingleOutputRuleEvaluationFactory>(l1RegularizationWeight,
                                                                          l2RegularizationWeight);
        std::unique_ptr<IDecomposableRuleEvaluationFactory> pruningRuleEvaluationFactoryPtr =
          std::make_unique<DecomposableSingleOutputRuleEvaluationFactory>(l1RegularizationWeight,
                                                                          l2RegularizationWeight);
        return std::make_unique<DenseDecomposableRegressionStatisticsProviderFactory<float64>>(
          std::move(lossFactoryPtr), std::move(evaluationMeasureFactoryPtr), std::move(defaultRuleEvaluationFactoryPtr),
          std::move(regularRuleEvaluationFactoryPtr), std::move(pruningRuleEvaluationFactoryPtr),
          multiThreadingSettings);
    }

    std::unique_ptr<IRegressionStatisticsProviderFactory>
      SingleOutputHeadConfig::createRegressionStatisticsProviderFactory(
        const IFeatureMatrix& featureMatrix, const IRowWiseRegressionMatrix& regressionMatrix,
        std::unique_ptr<INonDecomposableRegressionLossFactory<float64>>& lossFactoryPtr,
        std::unique_ptr<IRegressionEvaluationMeasureFactory<float64>>& evaluationMeasureFactoryPtr,
        const BlasFactory& blasFactory, const LapackFactory& lapackFactory) const {
        float32 l1RegularizationWeight = l1RegularizationConfig_.get().getWeight();
        float32 l2RegularizationWeight = l2RegularizationConfig_.get().getWeight();
        MultiThreadingSettings multiThreadingSettings =
          multiThreadingConfig_.get().getSettings(featureMatrix, regressionMatrix.getNumOutputs());
        std::unique_ptr<INonDecomposableRuleEvaluationFactory> defaultRuleEvaluationFactoryPtr =
          labelBinningConfig_.get().createNonDecomposableCompleteRuleEvaluationFactory(blasFactory, lapackFactory);
        std::unique_ptr<IDecomposableRuleEvaluationFactory> regularRuleEvaluationFactoryPtr =
          std::make_unique<DecomposableSingleOutputRuleEvaluationFactory>(l1RegularizationWeight,
                                                                          l2RegularizationWeight);
        std::unique_ptr<IDecomposableRuleEvaluationFactory> pruningRuleEvaluationFactoryPtr =
          std::make_unique<DecomposableSingleOutputRuleEvaluationFactory>(l1RegularizationWeight,
                                                                          l2RegularizationWeight);
        return std::make_unique<DenseConvertibleNonDecomposableRegressionStatisticsProviderFactory<float64>>(
          std::move(lossFactoryPtr), std::move(evaluationMeasureFactoryPtr), std::move(defaultRuleEvaluationFactoryPtr),
          std::move(regularRuleEvaluationFactoryPtr), std::move(pruningRuleEvaluationFactoryPtr),
          multiThreadingSettings);
    }

    bool SingleOutputHeadConfig::isPartial() const {
        return true;
    }

    bool SingleOutputHeadConfig::isSingleOutput() const {
        return true;
    }

}
