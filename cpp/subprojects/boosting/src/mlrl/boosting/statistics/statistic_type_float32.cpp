#include "mlrl/boosting/statistics/statistic_type_float32.hpp"

namespace boosting {

    Float32StatisticsConfig::Float32StatisticsConfig(ReadableProperty<IHeadConfig> headConfig)
        : headConfig_(headConfig) {}

    std::unique_ptr<IClassificationStatisticsProviderFactory>
      Float32StatisticsConfig::createClassificationStatisticsProviderFactory(
        const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix,
        const IDecomposableClassificationLossConfig& lossConfig) const {
        std::unique_ptr<IDecomposableClassificationLossFactory<float64>> lossFactoryPtr =
          lossConfig.createDecomposableClassificationLossFactory();
        std::unique_ptr<IClassificationEvaluationMeasureFactory<float64>> evaluationMeasureFactoryPtr =
          lossConfig.createClassificationEvaluationMeasureFactory();
        return headConfig_.get().create64BitPreset()->createClassificationStatisticsProviderFactory(
          featureMatrix, labelMatrix, lossFactoryPtr, evaluationMeasureFactoryPtr);
    }

    std::unique_ptr<IClassificationStatisticsProviderFactory>
      Float32StatisticsConfig::createClassificationStatisticsProviderFactory(
        const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix,
        const ISparseDecomposableClassificationLossConfig& lossConfig) const {
        std::unique_ptr<ISparseDecomposableClassificationLossFactory<float64>> lossFactoryPtr =
          lossConfig.createSparseDecomposableClassificationLossFactory();
        std::unique_ptr<ISparseEvaluationMeasureFactory<float64>> evaluationMeasureFactoryPtr =
          lossConfig.createSparseEvaluationMeasureFactory();
        return headConfig_.get().create64BitPreset()->createClassificationStatisticsProviderFactory(
          featureMatrix, labelMatrix, lossFactoryPtr, evaluationMeasureFactoryPtr);
    }

    std::unique_ptr<IClassificationStatisticsProviderFactory>
      Float32StatisticsConfig::createClassificationStatisticsProviderFactory(
        const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix,
        const INonDecomposableClassificationLossConfig& lossConfig, const BlasFactory& blasFactory,
        const LapackFactory& lapackFactory) const {
        std::unique_ptr<INonDecomposableClassificationLossFactory<float64>> lossFactoryPtr =
          lossConfig.createNonDecomposableClassificationLossFactory();
        std::unique_ptr<IClassificationEvaluationMeasureFactory<float64>> evaluationMeasureFactoryPtr =
          lossConfig.createClassificationEvaluationMeasureFactory();
        return headConfig_.get().create64BitPreset()->createClassificationStatisticsProviderFactory(
          featureMatrix, labelMatrix, lossFactoryPtr, evaluationMeasureFactoryPtr, blasFactory, lapackFactory);
    }

    std::unique_ptr<IRegressionStatisticsProviderFactory>
      Float32StatisticsConfig::createRegressionStatisticsProviderFactory(
        const IFeatureMatrix& featureMatrix, const IRowWiseRegressionMatrix& regressionMatrix,
        const IDecomposableRegressionLossConfig& lossConfig) const {
        std::unique_ptr<IDecomposableRegressionLossFactory<float64>> lossFactoryPtr =
          lossConfig.createDecomposableRegressionLossFactory();
        std::unique_ptr<IRegressionEvaluationMeasureFactory<float64>> evaluationMeasureFactoryPtr =
          lossConfig.createRegressionEvaluationMeasureFactory();
        return headConfig_.get().create64BitPreset()->createRegressionStatisticsProviderFactory(
          featureMatrix, regressionMatrix, lossFactoryPtr, evaluationMeasureFactoryPtr);
    }

    std::unique_ptr<IRegressionStatisticsProviderFactory>
      Float32StatisticsConfig::createRegressionStatisticsProviderFactory(
        const IFeatureMatrix& featureMatrix, const IRowWiseRegressionMatrix& regressionMatrix,
        const INonDecomposableRegressionLossConfig& lossConfig, const BlasFactory& blasFactory,
        const LapackFactory& lapackFactory) const {
        std::unique_ptr<INonDecomposableRegressionLossFactory<float64>> lossFactoryPtr =
          lossConfig.createNonDecomposableRegressionLossFactory();
        std::unique_ptr<IRegressionEvaluationMeasureFactory<float64>> evaluationMeasureFactoryPtr =
          lossConfig.createRegressionEvaluationMeasureFactory();
        return headConfig_.get().create64BitPreset()->createRegressionStatisticsProviderFactory(
          featureMatrix, regressionMatrix, lossFactoryPtr, evaluationMeasureFactoryPtr, blasFactory, lapackFactory);
    }
}
