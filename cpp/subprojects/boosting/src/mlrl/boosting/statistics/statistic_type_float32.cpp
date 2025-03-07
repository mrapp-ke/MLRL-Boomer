#include "mlrl/boosting/statistics/statistic_type_float32.hpp"

namespace boosting {

    Float32StatisticsConfig::Float32StatisticsConfig(ReadableProperty<IHeadConfig> headConfig)
        : headConfig_(headConfig) {}

    std::unique_ptr<IClassificationStatisticsProviderFactory>
      Float32StatisticsConfig::createClassificationStatisticsProviderFactory(
        const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix,
        const IDecomposableClassificationLossConfig& lossConfig) const {
        std::unique_ptr<IDecomposableClassificationLossConfig::IPreset<float32>> lossPresetPtr =
          lossConfig.createDecomposable32BitClassificationPreset();
        std::unique_ptr<IDecomposableClassificationLossFactory<float32>> lossFactoryPtr =
          lossPresetPtr->createDecomposableClassificationLossFactory();
        std::unique_ptr<IClassificationEvaluationMeasureFactory<float32>> evaluationMeasureFactoryPtr =
          lossPresetPtr->createClassificationEvaluationMeasureFactory();
        return headConfig_.get().create32BitPreset()->createClassificationStatisticsProviderFactory(
          featureMatrix, labelMatrix, lossFactoryPtr, evaluationMeasureFactoryPtr);
    }

    std::unique_ptr<IClassificationStatisticsProviderFactory>
      Float32StatisticsConfig::createClassificationStatisticsProviderFactory(
        const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix,
        const ISparseDecomposableClassificationLossConfig& lossConfig) const {
        std::unique_ptr<ISparseDecomposableClassificationLossConfig::IPreset<float32>> lossPresetPtr =
          lossConfig.createSparseDecomposable32BitClassificationPreset();
        std::unique_ptr<ISparseDecomposableClassificationLossFactory<float32>> lossFactoryPtr =
          lossPresetPtr->createSparseDecomposableClassificationLossFactory();
        std::unique_ptr<ISparseEvaluationMeasureFactory<float32>> evaluationMeasureFactoryPtr =
          lossPresetPtr->createSparseEvaluationMeasureFactory();
        return headConfig_.get().create32BitPreset()->createClassificationStatisticsProviderFactory(
          featureMatrix, labelMatrix, lossFactoryPtr, evaluationMeasureFactoryPtr);
    }

    std::unique_ptr<IClassificationStatisticsProviderFactory>
      Float32StatisticsConfig::createClassificationStatisticsProviderFactory(
        const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix,
        const INonDecomposableClassificationLossConfig& lossConfig, const BlasFactory& blasFactory,
        const LapackFactory& lapackFactory) const {
        std::unique_ptr<INonDecomposableClassificationLossConfig::IPreset<float32>> lossPresetPtr =
          lossConfig.createNonDecomposable32BitClassificationPreset();
        std::unique_ptr<INonDecomposableClassificationLossFactory<float32>> lossFactoryPtr =
          lossPresetPtr->createNonDecomposableClassificationLossFactory();
        std::unique_ptr<IClassificationEvaluationMeasureFactory<float32>> evaluationMeasureFactoryPtr =
          lossPresetPtr->createClassificationEvaluationMeasureFactory();
        return headConfig_.get().create32BitPreset()->createClassificationStatisticsProviderFactory(
          featureMatrix, labelMatrix, lossFactoryPtr, evaluationMeasureFactoryPtr, blasFactory, lapackFactory);
    }

    std::unique_ptr<IRegressionStatisticsProviderFactory>
      Float32StatisticsConfig::createRegressionStatisticsProviderFactory(
        const IFeatureMatrix& featureMatrix, const IRowWiseRegressionMatrix& regressionMatrix,
        const IDecomposableRegressionLossConfig& lossConfig) const {
        std::unique_ptr<IDecomposableRegressionLossConfig::IPreset<float32>> lossPresetPtr =
          lossConfig.createDecomposable32BitRegressionPreset();
        std::unique_ptr<IDecomposableRegressionLossFactory<float32>> lossFactoryPtr =
          lossPresetPtr->createDecomposableRegressionLossFactory();
        std::unique_ptr<IRegressionEvaluationMeasureFactory<float32>> evaluationMeasureFactoryPtr =
          lossPresetPtr->createRegressionEvaluationMeasureFactory();
        return headConfig_.get().create32BitPreset()->createRegressionStatisticsProviderFactory(
          featureMatrix, regressionMatrix, lossFactoryPtr, evaluationMeasureFactoryPtr);
    }

    std::unique_ptr<IRegressionStatisticsProviderFactory>
      Float32StatisticsConfig::createRegressionStatisticsProviderFactory(
        const IFeatureMatrix& featureMatrix, const IRowWiseRegressionMatrix& regressionMatrix,
        const INonDecomposableRegressionLossConfig& lossConfig, const BlasFactory& blasFactory,
        const LapackFactory& lapackFactory) const {
        std::unique_ptr<INonDecomposableRegressionLossConfig::IPreset<float32>> lossPresetPtr =
          lossConfig.createNonDecomposable32BitRegressionPreset();
        std::unique_ptr<INonDecomposableRegressionLossFactory<float32>> lossFactoryPtr =
          lossPresetPtr->createNonDecomposableRegressionLossFactory();
        std::unique_ptr<IRegressionEvaluationMeasureFactory<float32>> evaluationMeasureFactoryPtr =
          lossPresetPtr->createRegressionEvaluationMeasureFactory();
        return headConfig_.get().create32BitPreset()->createRegressionStatisticsProviderFactory(
          featureMatrix, regressionMatrix, lossFactoryPtr, evaluationMeasureFactoryPtr, blasFactory, lapackFactory);
    }
}
