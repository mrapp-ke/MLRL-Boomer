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
        const IDecomposableClassificationLossConfig& lossConfig) const {
        float32 l1RegularizationWeight = l1RegularizationConfig_.get().getWeight();
        float32 l2RegularizationWeight = l2RegularizationConfig_.get().getWeight();
        MultiThreadingSettings multiThreadingSettings =
          multiThreadingConfig_.get().getSettings(featureMatrix, labelMatrix.getNumOutputs());
        std::unique_ptr<IDecomposableClassificationLossFactory> lossFactoryPtr =
          lossConfig.createDecomposableClassificationLossFactory();
        std::unique_ptr<IClassificationEvaluationMeasureFactory> evaluationMeasureFactoryPtr =
          lossConfig.createClassificationEvaluationMeasureFactory();
        std::unique_ptr<IDecomposableRuleEvaluationFactory> defaultRuleEvaluationFactoryPtr =
          labelBinningConfig_.get().createDecomposableCompleteRuleEvaluationFactory();
        std::unique_ptr<IDecomposableRuleEvaluationFactory> regularRuleEvaluationFactoryPtr =
          std::make_unique<DecomposableSingleOutputRuleEvaluationFactory>(l1RegularizationWeight,
                                                                          l2RegularizationWeight);
        std::unique_ptr<IDecomposableRuleEvaluationFactory> pruningRuleEvaluationFactoryPtr =
          std::make_unique<DecomposableSingleOutputRuleEvaluationFactory>(l1RegularizationWeight,
                                                                          l2RegularizationWeight);
        return std::make_unique<DenseDecomposableClassificationStatisticsProviderFactory>(
          std::move(lossFactoryPtr), std::move(evaluationMeasureFactoryPtr), std::move(defaultRuleEvaluationFactoryPtr),
          std::move(regularRuleEvaluationFactoryPtr), std::move(pruningRuleEvaluationFactoryPtr),
          multiThreadingSettings);
    }

    std::unique_ptr<IClassificationStatisticsProviderFactory>
      SingleOutputHeadConfig::createClassificationStatisticsProviderFactory(
        const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix,
        const ISparseDecomposableClassificationLossConfig& lossConfig) const {
        float32 l1RegularizationWeight = l1RegularizationConfig_.get().getWeight();
        float32 l2RegularizationWeight = l2RegularizationConfig_.get().getWeight();
        MultiThreadingSettings multiThreadingSettings =
          multiThreadingConfig_.get().getSettings(featureMatrix, labelMatrix.getNumOutputs());
        std::unique_ptr<ISparseDecomposableClassificationLossFactory> lossFactoryPtr =
          lossConfig.createSparseDecomposableClassificationLossFactory();
        std::unique_ptr<ISparseEvaluationMeasureFactory> evaluationMeasureFactoryPtr =
          lossConfig.createSparseEvaluationMeasureFactory();
        std::unique_ptr<ISparseDecomposableRuleEvaluationFactory> regularRuleEvaluationFactoryPtr =
          std::make_unique<DecomposableSingleOutputRuleEvaluationFactory>(l1RegularizationWeight,
                                                                          l2RegularizationWeight);
        std::unique_ptr<ISparseDecomposableRuleEvaluationFactory> pruningRuleEvaluationFactoryPtr =
          std::make_unique<DecomposableSingleOutputRuleEvaluationFactory>(l1RegularizationWeight,
                                                                          l2RegularizationWeight);
        return std::make_unique<SparseDecomposableClassificationStatisticsProviderFactory>(
          std::move(lossFactoryPtr), std::move(evaluationMeasureFactoryPtr), std::move(regularRuleEvaluationFactoryPtr),
          std::move(pruningRuleEvaluationFactoryPtr), multiThreadingSettings);
    }

    std::unique_ptr<IClassificationStatisticsProviderFactory>
      SingleOutputHeadConfig::createClassificationStatisticsProviderFactory(
        const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix,
        const INonDecomposableClassificationLossConfig& lossConfig, const Blas& blas, const Lapack& lapack) const {
        float32 l1RegularizationWeight = l1RegularizationConfig_.get().getWeight();
        float32 l2RegularizationWeight = l2RegularizationConfig_.get().getWeight();
        MultiThreadingSettings multiThreadingSettings =
          multiThreadingConfig_.get().getSettings(featureMatrix, labelMatrix.getNumOutputs());
        std::unique_ptr<INonDecomposableClassificationLossFactory> lossFactoryPtr =
          lossConfig.createNonDecomposableClassificationLossFactory();
        std::unique_ptr<IClassificationEvaluationMeasureFactory> evaluationMeasureFactoryPtr =
          lossConfig.createClassificationEvaluationMeasureFactory();
        std::unique_ptr<INonDecomposableRuleEvaluationFactory> defaultRuleEvaluationFactoryPtr =
          labelBinningConfig_.get().createNonDecomposableCompleteRuleEvaluationFactory(blas, lapack);
        std::unique_ptr<IDecomposableRuleEvaluationFactory> regularRuleEvaluationFactoryPtr =
          std::make_unique<DecomposableSingleOutputRuleEvaluationFactory>(l1RegularizationWeight,
                                                                          l2RegularizationWeight);
        std::unique_ptr<IDecomposableRuleEvaluationFactory> pruningRuleEvaluationFactoryPtr =
          std::make_unique<DecomposableSingleOutputRuleEvaluationFactory>(l1RegularizationWeight,
                                                                          l2RegularizationWeight);
        return std::make_unique<DenseConvertibleNonDecomposableClassificationStatisticsProviderFactory>(
          std::move(lossFactoryPtr), std::move(evaluationMeasureFactoryPtr), std::move(defaultRuleEvaluationFactoryPtr),
          std::move(regularRuleEvaluationFactoryPtr), std::move(pruningRuleEvaluationFactoryPtr),
          multiThreadingSettings);
    }

    std::unique_ptr<IRegressionStatisticsProviderFactory>
      SingleOutputHeadConfig::createRegressionStatisticsProviderFactory(
        const IFeatureMatrix& featureMatrix, const IRowWiseRegressionMatrix& regressionMatrix,
        const IDecomposableRegressionLossConfig& lossConfig) const {
        float32 l1RegularizationWeight = l1RegularizationConfig_.get().getWeight();
        float32 l2RegularizationWeight = l2RegularizationConfig_.get().getWeight();
        MultiThreadingSettings multiThreadingSettings =
          multiThreadingConfig_.get().getSettings(featureMatrix, regressionMatrix.getNumOutputs());
        std::unique_ptr<IDecomposableRegressionLossFactory> lossFactoryPtr =
          lossConfig.createDecomposableRegressionLossFactory();
        std::unique_ptr<IRegressionEvaluationMeasureFactory> evaluationMeasureFactoryPtr =
          lossConfig.createRegressionEvaluationMeasureFactory();
        std::unique_ptr<IDecomposableRuleEvaluationFactory> defaultRuleEvaluationFactoryPtr =
          labelBinningConfig_.get().createDecomposableCompleteRuleEvaluationFactory();
        std::unique_ptr<IDecomposableRuleEvaluationFactory> regularRuleEvaluationFactoryPtr =
          std::make_unique<DecomposableSingleOutputRuleEvaluationFactory>(l1RegularizationWeight,
                                                                          l2RegularizationWeight);
        std::unique_ptr<IDecomposableRuleEvaluationFactory> pruningRuleEvaluationFactoryPtr =
          std::make_unique<DecomposableSingleOutputRuleEvaluationFactory>(l1RegularizationWeight,
                                                                          l2RegularizationWeight);
        return std::make_unique<DenseDecomposableRegressionStatisticsProviderFactory>(
          std::move(lossFactoryPtr), std::move(evaluationMeasureFactoryPtr), std::move(defaultRuleEvaluationFactoryPtr),
          std::move(regularRuleEvaluationFactoryPtr), std::move(pruningRuleEvaluationFactoryPtr),
          multiThreadingSettings);
    }

    std::unique_ptr<IRegressionStatisticsProviderFactory>
      SingleOutputHeadConfig::createRegressionStatisticsProviderFactory(
        const IFeatureMatrix& featureMatrix, const IRowWiseRegressionMatrix& regressionMatrix,
        const INonDecomposableRegressionLossConfig& lossConfig, const Blas& blas, const Lapack& lapack) const {
        float32 l1RegularizationWeight = l1RegularizationConfig_.get().getWeight();
        float32 l2RegularizationWeight = l2RegularizationConfig_.get().getWeight();
        MultiThreadingSettings multiThreadingSettings =
          multiThreadingConfig_.get().getSettings(featureMatrix, regressionMatrix.getNumOutputs());
        std::unique_ptr<INonDecomposableRegressionLossFactory> lossFactoryPtr =
          lossConfig.createNonDecomposableRegressionLossFactory();
        std::unique_ptr<IRegressionEvaluationMeasureFactory> evaluationMeasureFactoryPtr =
          lossConfig.createRegressionEvaluationMeasureFactory();
        std::unique_ptr<INonDecomposableRuleEvaluationFactory> defaultRuleEvaluationFactoryPtr =
          labelBinningConfig_.get().createNonDecomposableCompleteRuleEvaluationFactory(blas, lapack);
        std::unique_ptr<IDecomposableRuleEvaluationFactory> regularRuleEvaluationFactoryPtr =
          std::make_unique<DecomposableSingleOutputRuleEvaluationFactory>(l1RegularizationWeight,
                                                                          l2RegularizationWeight);
        std::unique_ptr<IDecomposableRuleEvaluationFactory> pruningRuleEvaluationFactoryPtr =
          std::make_unique<DecomposableSingleOutputRuleEvaluationFactory>(l1RegularizationWeight,
                                                                          l2RegularizationWeight);
        return std::make_unique<DenseConvertibleNonDecomposableRegressionStatisticsProviderFactory>(
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
