#include "mlrl/boosting/rule_evaluation/head_type_complete.hpp"

#include "mlrl/boosting/rule_evaluation/rule_evaluation_decomposable_complete.hpp"
#include "mlrl/boosting/statistics/statistics_provider_decomposable_dense.hpp"
#include "mlrl/boosting/statistics/statistics_provider_non_decomposable_dense.hpp"

namespace boosting {

    CompleteHeadConfig::CompleteHeadConfig(GetterFunction<ILabelBinningConfig> labelBinningConfigGetter,
                                           GetterFunction<IMultiThreadingConfig> multiThreadingConfigGetter,
                                           GetterFunction<IRegularizationConfig> l1RegularizationConfigGetter,
                                           GetterFunction<IRegularizationConfig> l2RegularizationConfigGetter)
        : labelBinningConfigGetter_(labelBinningConfigGetter), multiThreadingConfigGetter_(multiThreadingConfigGetter),
          l1RegularizationConfigGetter_(l1RegularizationConfigGetter),
          l2RegularizationConfigGetter_(l2RegularizationConfigGetter) {}

    std::unique_ptr<IStatisticsProviderFactory> CompleteHeadConfig::createStatisticsProviderFactory(
      const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix,
      const IDecomposableLossConfig& lossConfig) const {
        float64 l1RegularizationWeight = l1RegularizationConfigGetter_().getWeight();
        float64 l2RegularizationWeight = l2RegularizationConfigGetter_().getWeight();
        uint32 numThreads = multiThreadingConfigGetter_().getNumThreads(featureMatrix, labelMatrix.getNumOutputs());
        std::unique_ptr<IDecomposableLossFactory> lossFactoryPtr = lossConfig.createDecomposableLossFactory();
        std::unique_ptr<IEvaluationMeasureFactory> evaluationMeasureFactoryPtr =
          lossConfig.createEvaluationMeasureFactory();
        std::unique_ptr<IDecomposableRuleEvaluationFactory> defaultRuleEvaluationFactoryPtr =
          labelBinningConfigGetter_().createDecomposableCompleteRuleEvaluationFactory();
        std::unique_ptr<IDecomposableRuleEvaluationFactory> regularRuleEvaluationFactoryPtr =
          std::make_unique<DecomposableCompleteRuleEvaluationFactory>(l1RegularizationWeight, l2RegularizationWeight);
        std::unique_ptr<IDecomposableRuleEvaluationFactory> pruningRuleEvaluationFactoryPtr =
          std::make_unique<DecomposableCompleteRuleEvaluationFactory>(l1RegularizationWeight, l2RegularizationWeight);
        return std::make_unique<DenseDecomposableStatisticsProviderFactory>(
          std::move(lossFactoryPtr), std::move(evaluationMeasureFactoryPtr), std::move(defaultRuleEvaluationFactoryPtr),
          std::move(regularRuleEvaluationFactoryPtr), std::move(pruningRuleEvaluationFactoryPtr), numThreads);
    }

    std::unique_ptr<IStatisticsProviderFactory> CompleteHeadConfig::createStatisticsProviderFactory(
      const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix,
      const ISparseDecomposableLossConfig& lossConfig) const {
        return this->createStatisticsProviderFactory(featureMatrix, labelMatrix,
                                                     static_cast<const IDecomposableLossConfig&>(lossConfig));
    }

    std::unique_ptr<IStatisticsProviderFactory> CompleteHeadConfig::createStatisticsProviderFactory(
      const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix,
      const INonDecomposableLossConfig& lossConfig, const Blas& blas, const Lapack& lapack) const {
        uint32 numThreads = multiThreadingConfigGetter_().getNumThreads(featureMatrix, labelMatrix.getNumOutputs());
        std::unique_ptr<INonDecomposableLossFactory> lossFactoryPtr = lossConfig.createNonDecomposableLossFactory();
        std::unique_ptr<IEvaluationMeasureFactory> evaluationMeasureFactoryPtr =
          lossConfig.createNonDecomposableLossFactory();
        std::unique_ptr<INonDecomposableRuleEvaluationFactory> defaultRuleEvaluationFactoryPtr =
          labelBinningConfigGetter_().createNonDecomposableCompleteRuleEvaluationFactory(blas, lapack);
        std::unique_ptr<INonDecomposableRuleEvaluationFactory> regularRuleEvaluationFactoryPtr =
          labelBinningConfigGetter_().createNonDecomposableCompleteRuleEvaluationFactory(blas, lapack);
        std::unique_ptr<INonDecomposableRuleEvaluationFactory> pruningRuleEvaluationFactoryPtr =
          labelBinningConfigGetter_().createNonDecomposableCompleteRuleEvaluationFactory(blas, lapack);
        return std::make_unique<DenseNonDecomposableStatisticsProviderFactory>(
          std::move(lossFactoryPtr), std::move(evaluationMeasureFactoryPtr), std::move(defaultRuleEvaluationFactoryPtr),
          std::move(regularRuleEvaluationFactoryPtr), std::move(pruningRuleEvaluationFactoryPtr), numThreads);
    }

    bool CompleteHeadConfig::isPartial() const {
        return false;
    }

    bool CompleteHeadConfig::isSingleOutput() const {
        return false;
    }

}
