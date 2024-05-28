#include "mlrl/boosting/rule_evaluation/head_type_partial_fixed.hpp"

#include "mlrl/boosting/statistics/statistics_provider_decomposable_dense.hpp"
#include "mlrl/boosting/statistics/statistics_provider_decomposable_sparse.hpp"
#include "mlrl/boosting/statistics/statistics_provider_non_decomposable_dense.hpp"
#include "mlrl/common/util/validation.hpp"

namespace boosting {

    static inline float32 calculateOutputRatio(float32 outputRatio, const IRowWiseLabelMatrix& labelMatrix) {
        if (outputRatio > 0) {
            return outputRatio;
        } else {
            return labelMatrix.calculateLabelCardinality() / labelMatrix.getNumLabels();
        }
    }

    FixedPartialHeadConfig::FixedPartialHeadConfig(
      const std::unique_ptr<ILabelBinningConfig>& labelBinningConfigPtr,
      const std::unique_ptr<IMultiThreadingConfig>& multiThreadingConfigPtr)
        : outputRatio_(0.0f), minOutputs_(2), maxOutputs_(0), labelBinningConfigPtr_(labelBinningConfigPtr),
          multiThreadingConfigPtr_(multiThreadingConfigPtr) {}

    float32 FixedPartialHeadConfig::getOutputRatio() const {
        return outputRatio_;
    }

    IFixedPartialHeadConfig& FixedPartialHeadConfig::setOutputRatio(float32 outputRatio) {
        if (!isEqualToZero(outputRatio)) {
            assertGreater<float32>("outputRatio", outputRatio, 0);
            assertLess<float32>("outputRatio", outputRatio, 1);
        }

        outputRatio_ = outputRatio;
        return *this;
    }

    uint32 FixedPartialHeadConfig::getMinOutputs() const {
        return minOutputs_;
    }

    IFixedPartialHeadConfig& FixedPartialHeadConfig::setMinOutputs(uint32 minOutputs) {
        assertGreaterOrEqual<uint32>("minOutputs", minOutputs, 2);
        minOutputs_ = minOutputs;
        return *this;
    }

    uint32 FixedPartialHeadConfig::getMaxOutputs() const {
        return maxOutputs_;
    }

    IFixedPartialHeadConfig& FixedPartialHeadConfig::setMaxOutputs(uint32 maxOutputs) {
        if (maxOutputs != 0) assertGreaterOrEqual<uint32>("maxOutputs", maxOutputs, minOutputs_);
        maxOutputs_ = maxOutputs;
        return *this;
    }

    std::unique_ptr<IStatisticsProviderFactory> FixedPartialHeadConfig::createStatisticsProviderFactory(
      const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix,
      const ILabelWiseLossConfig& lossConfig) const {
        uint32 numThreads = multiThreadingConfigPtr_->getNumThreads(featureMatrix, labelMatrix.getNumLabels());
        float32 outputRatio = calculateOutputRatio(outputRatio_, labelMatrix);
        std::unique_ptr<ILabelWiseLossFactory> lossFactoryPtr = lossConfig.createLabelWiseLossFactory();
        std::unique_ptr<IEvaluationMeasureFactory> evaluationMeasureFactoryPtr =
          lossConfig.createEvaluationMeasureFactory();
        std::unique_ptr<IDecomposableRuleEvaluationFactory> defaultRuleEvaluationFactoryPtr =
          labelBinningConfigPtr_->createDecomposableCompleteRuleEvaluationFactory();
        std::unique_ptr<IDecomposableRuleEvaluationFactory> regularRuleEvaluationFactoryPtr =
          labelBinningConfigPtr_->createDecomposableFixedPartialRuleEvaluationFactory(outputRatio, minOutputs_,
                                                                                      maxOutputs_);
        std::unique_ptr<IDecomposableRuleEvaluationFactory> pruningRuleEvaluationFactoryPtr =
          labelBinningConfigPtr_->createDecomposableFixedPartialRuleEvaluationFactory(outputRatio, minOutputs_,
                                                                                      maxOutputs_);
        return std::make_unique<DenseDecomposableStatisticsProviderFactory>(
          std::move(lossFactoryPtr), std::move(evaluationMeasureFactoryPtr), std::move(defaultRuleEvaluationFactoryPtr),
          std::move(regularRuleEvaluationFactoryPtr), std::move(pruningRuleEvaluationFactoryPtr), numThreads);
    }

    std::unique_ptr<IStatisticsProviderFactory> FixedPartialHeadConfig::createStatisticsProviderFactory(
      const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix,
      const ISparseLabelWiseLossConfig& lossConfig) const {
        uint32 numThreads = multiThreadingConfigPtr_->getNumThreads(featureMatrix, labelMatrix.getNumLabels());
        float32 outputRatio = calculateOutputRatio(outputRatio_, labelMatrix);
        std::unique_ptr<ISparseLabelWiseLossFactory> lossFactoryPtr = lossConfig.createSparseLabelWiseLossFactory();
        std::unique_ptr<ISparseEvaluationMeasureFactory> evaluationMeasureFactoryPtr =
          lossConfig.createSparseEvaluationMeasureFactory();
        std::unique_ptr<ISparseDecomposableRuleEvaluationFactory> regularRuleEvaluationFactoryPtr =
          labelBinningConfigPtr_->createDecomposableFixedPartialRuleEvaluationFactory(outputRatio, minOutputs_,
                                                                                      maxOutputs_);
        std::unique_ptr<ISparseDecomposableRuleEvaluationFactory> pruningRuleEvaluationFactoryPtr =
          labelBinningConfigPtr_->createDecomposableFixedPartialRuleEvaluationFactory(outputRatio, minOutputs_,
                                                                                      maxOutputs_);
        return std::make_unique<SparseDecomposableStatisticsProviderFactory>(
          std::move(lossFactoryPtr), std::move(evaluationMeasureFactoryPtr), std::move(regularRuleEvaluationFactoryPtr),
          std::move(pruningRuleEvaluationFactoryPtr), numThreads);
    }

    std::unique_ptr<IStatisticsProviderFactory> FixedPartialHeadConfig::createStatisticsProviderFactory(
      const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix,
      const IExampleWiseLossConfig& lossConfig, const Blas& blas, const Lapack& lapack) const {
        uint32 numThreads = multiThreadingConfigPtr_->getNumThreads(featureMatrix, labelMatrix.getNumLabels());
        float32 outputRatio = calculateOutputRatio(outputRatio_, labelMatrix);
        std::unique_ptr<IExampleWiseLossFactory> lossFactoryPtr = lossConfig.createExampleWiseLossFactory();
        std::unique_ptr<IEvaluationMeasureFactory> evaluationMeasureFactoryPtr =
          lossConfig.createExampleWiseLossFactory();
        std::unique_ptr<INonDecomposableRuleEvaluationFactory> defaultRuleEvaluationFactoryPtr =
          labelBinningConfigPtr_->createNonDecomposableCompleteRuleEvaluationFactory(blas, lapack);
        std::unique_ptr<INonDecomposableRuleEvaluationFactory> regularRuleEvaluationFactoryPtr =
          labelBinningConfigPtr_->createNonDecomposableFixedPartialRuleEvaluationFactory(outputRatio, minOutputs_,
                                                                                         maxOutputs_, blas, lapack);
        std::unique_ptr<INonDecomposableRuleEvaluationFactory> pruningRuleEvaluationFactoryPtr =
          labelBinningConfigPtr_->createNonDecomposableFixedPartialRuleEvaluationFactory(outputRatio, minOutputs_,
                                                                                         maxOutputs_, blas, lapack);
        return std::make_unique<DenseNonDecomposableStatisticsProviderFactory>(
          std::move(lossFactoryPtr), std::move(evaluationMeasureFactoryPtr), std::move(defaultRuleEvaluationFactoryPtr),
          std::move(regularRuleEvaluationFactoryPtr), std::move(pruningRuleEvaluationFactoryPtr), numThreads);
    }

    bool FixedPartialHeadConfig::isPartial() const {
        return true;
    }

    bool FixedPartialHeadConfig::isSingleOutput() const {
        return false;
    }

}
