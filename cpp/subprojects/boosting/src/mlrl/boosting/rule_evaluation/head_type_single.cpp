#include "mlrl/boosting/rule_evaluation/head_type_single.hpp"

#include "mlrl/boosting/rule_evaluation/rule_evaluation_decomposable_single.hpp"
#include "mlrl/boosting/statistics/statistics_provider_decomposable_dense.hpp"
#include "mlrl/boosting/statistics/statistics_provider_decomposable_sparse.hpp"
#include "mlrl/boosting/statistics/statistics_provider_non_decomposable_dense.hpp"

namespace boosting {

    SingleOutputHeadConfig::SingleOutputHeadConfig(
      const std::unique_ptr<ILabelBinningConfig>& labelBinningConfigPtr,
      const std::unique_ptr<IMultiThreadingConfig>& multiThreadingConfigPtr,
      const std::unique_ptr<IRegularizationConfig>& l1RegularizationConfigPtr,
      const std::unique_ptr<IRegularizationConfig>& l2RegularizationConfigPtr)
        : labelBinningConfigPtr_(labelBinningConfigPtr), multiThreadingConfigPtr_(multiThreadingConfigPtr),
          l1RegularizationConfigPtr_(l1RegularizationConfigPtr), l2RegularizationConfigPtr_(l2RegularizationConfigPtr) {

    }

    std::unique_ptr<IStatisticsProviderFactory> SingleOutputHeadConfig::createStatisticsProviderFactory(
      const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix,
      const IDecomposableLossConfig& lossConfig) const {
        float64 l1RegularizationWeight = l1RegularizationConfigPtr_->getWeight();
        float64 l2RegularizationWeight = l2RegularizationConfigPtr_->getWeight();
        uint32 numThreads = multiThreadingConfigPtr_->getNumThreads(featureMatrix, labelMatrix.getNumOutputs());
        std::unique_ptr<IDecomposableLossFactory> lossFactoryPtr = lossConfig.createDecomposableLossFactory();
        std::unique_ptr<IEvaluationMeasureFactory> evaluationMeasureFactoryPtr =
          lossConfig.createEvaluationMeasureFactory();
        std::unique_ptr<IDecomposableRuleEvaluationFactory> defaultRuleEvaluationFactoryPtr =
          labelBinningConfigPtr_->createDecomposableCompleteRuleEvaluationFactory();
        std::unique_ptr<IDecomposableRuleEvaluationFactory> regularRuleEvaluationFactoryPtr =
          std::make_unique<DecomposableSingleOutputRuleEvaluationFactory>(l1RegularizationWeight,
                                                                          l2RegularizationWeight);
        std::unique_ptr<IDecomposableRuleEvaluationFactory> pruningRuleEvaluationFactoryPtr =
          std::make_unique<DecomposableSingleOutputRuleEvaluationFactory>(l1RegularizationWeight,
                                                                          l2RegularizationWeight);
        return std::make_unique<DenseDecomposableStatisticsProviderFactory>(
          std::move(lossFactoryPtr), std::move(evaluationMeasureFactoryPtr), std::move(defaultRuleEvaluationFactoryPtr),
          std::move(regularRuleEvaluationFactoryPtr), std::move(pruningRuleEvaluationFactoryPtr), numThreads);
    }

    std::unique_ptr<IStatisticsProviderFactory> SingleOutputHeadConfig::createStatisticsProviderFactory(
      const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix,
      const ISparseDecomposableLossConfig& lossConfig) const {
        float64 l1RegularizationWeight = l1RegularizationConfigPtr_->getWeight();
        float64 l2RegularizationWeight = l2RegularizationConfigPtr_->getWeight();
        uint32 numThreads = multiThreadingConfigPtr_->getNumThreads(featureMatrix, labelMatrix.getNumOutputs());
        std::unique_ptr<ISparseDecomposableLossFactory> lossFactoryPtr =
          lossConfig.createSparseDecomposableLossFactory();
        std::unique_ptr<ISparseEvaluationMeasureFactory> evaluationMeasureFactoryPtr =
          lossConfig.createSparseEvaluationMeasureFactory();
        std::unique_ptr<ISparseDecomposableRuleEvaluationFactory> regularRuleEvaluationFactoryPtr =
          std::make_unique<DecomposableSingleOutputRuleEvaluationFactory>(l1RegularizationWeight,
                                                                          l2RegularizationWeight);
        std::unique_ptr<ISparseDecomposableRuleEvaluationFactory> pruningRuleEvaluationFactoryPtr =
          std::make_unique<DecomposableSingleOutputRuleEvaluationFactory>(l1RegularizationWeight,
                                                                          l2RegularizationWeight);
        return std::make_unique<SparseDecomposableStatisticsProviderFactory>(
          std::move(lossFactoryPtr), std::move(evaluationMeasureFactoryPtr), std::move(regularRuleEvaluationFactoryPtr),
          std::move(pruningRuleEvaluationFactoryPtr), numThreads);
    }

    std::unique_ptr<IStatisticsProviderFactory> SingleOutputHeadConfig::createStatisticsProviderFactory(
      const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix,
      const INonDecomposableLossConfig& lossConfig, const Blas& blas, const Lapack& lapack) const {
        float64 l1RegularizationWeight = l1RegularizationConfigPtr_->getWeight();
        float64 l2RegularizationWeight = l2RegularizationConfigPtr_->getWeight();
        uint32 numThreads = multiThreadingConfigPtr_->getNumThreads(featureMatrix, labelMatrix.getNumOutputs());
        std::unique_ptr<INonDecomposableLossFactory> lossFactoryPtr = lossConfig.createNonDecomposableLossFactory();
        std::unique_ptr<IEvaluationMeasureFactory> evaluationMeasureFactoryPtr =
          lossConfig.createNonDecomposableLossFactory();
        std::unique_ptr<INonDecomposableRuleEvaluationFactory> defaultRuleEvaluationFactoryPtr =
          labelBinningConfigPtr_->createNonDecomposableCompleteRuleEvaluationFactory(blas, lapack);
        std::unique_ptr<IDecomposableRuleEvaluationFactory> regularRuleEvaluationFactoryPtr =
          std::make_unique<DecomposableSingleOutputRuleEvaluationFactory>(l1RegularizationWeight,
                                                                          l2RegularizationWeight);
        std::unique_ptr<IDecomposableRuleEvaluationFactory> pruningRuleEvaluationFactoryPtr =
          std::make_unique<DecomposableSingleOutputRuleEvaluationFactory>(l1RegularizationWeight,
                                                                          l2RegularizationWeight);
        return std::make_unique<DenseConvertibleNonDecomposableStatisticsProviderFactory>(
          std::move(lossFactoryPtr), std::move(evaluationMeasureFactoryPtr), std::move(defaultRuleEvaluationFactoryPtr),
          std::move(regularRuleEvaluationFactoryPtr), std::move(pruningRuleEvaluationFactoryPtr), numThreads);
    }

    bool SingleOutputHeadConfig::isPartial() const {
        return true;
    }

    bool SingleOutputHeadConfig::isSingleOutput() const {
        return true;
    }

}
