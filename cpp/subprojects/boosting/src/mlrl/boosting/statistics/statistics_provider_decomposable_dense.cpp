#ifdef _WIN32
    #pragma warning(push)
    #pragma warning(disable : 4250)
#endif

#include "mlrl/boosting/statistics/statistics_provider_decomposable_dense.hpp"

#include "mlrl/common/util/openmp.hpp"
#include "statistics_decomposable_dense.hpp"
#include "statistics_provider_decomposable.hpp"

namespace boosting {

    template<typename Loss, typename OutputMatrix, typename EvaluationMeasure>
    static inline std::unique_ptr<IDecomposableStatistics<IDecomposableRuleEvaluationFactory>> createStatistics(
      std::unique_ptr<Loss> lossPtr, std::unique_ptr<EvaluationMeasure> evaluationMeasurePtr,
      const IDecomposableRuleEvaluationFactory& ruleEvaluationFactory, uint32 numThreads,
      const OutputMatrix& outputMatrix) {
        uint32 numExamples = outputMatrix.numRows;
        uint32 numOutputs = outputMatrix.numCols;
        std::unique_ptr<DenseDecomposableStatisticMatrix> statisticMatrixPtr =
          std::make_unique<DenseDecomposableStatisticMatrix>(numExamples, numOutputs);
        std::unique_ptr<NumericCContiguousMatrix<float64>> scoreMatrixPtr =
          std::make_unique<NumericCContiguousMatrix<float64>>(numExamples, numOutputs, true);
        const Loss* lossRawPtr = lossPtr.get();
        const OutputMatrix* outputMatrixPtr = &outputMatrix;
        const CContiguousView<float64>* scoreMatrixRawPtr = &scoreMatrixPtr->getView();
        CContiguousView<Tuple<float64>>* statisticMatrixRawPtr = &statisticMatrixPtr->getView();

#if MULTI_THREADING_SUPPORT_ENABLED
    #pragma omp parallel for firstprivate(numExamples) firstprivate(lossRawPtr) firstprivate(outputMatrixPtr) \
      firstprivate(scoreMatrixRawPtr) firstprivate(statisticMatrixRawPtr) schedule(dynamic) num_threads(numThreads)
#endif
        for (int64 i = 0; i < numExamples; i++) {
            lossRawPtr->updateDecomposableStatistics(i, *outputMatrixPtr, *scoreMatrixRawPtr, IndexIterator(),
                                                     IndexIterator(outputMatrixPtr->numCols), *statisticMatrixRawPtr);
        }

        return std::make_unique<DenseDecomposableStatistics<Loss, OutputMatrix, EvaluationMeasure>>(
          std::move(lossPtr), std::move(evaluationMeasurePtr), ruleEvaluationFactory, outputMatrix,
          std::move(statisticMatrixPtr), std::move(scoreMatrixPtr));
    }

    DenseDecomposableClassificationStatisticsProviderFactory::DenseDecomposableClassificationStatisticsProviderFactory(
      std::unique_ptr<IDecomposableClassificationLossFactory> lossFactoryPtr,
      std::unique_ptr<IClassificationEvaluationMeasureFactory> evaluationMeasureFactoryPtr,
      std::unique_ptr<IDecomposableRuleEvaluationFactory> defaultRuleEvaluationFactoryPtr,
      std::unique_ptr<IDecomposableRuleEvaluationFactory> regularRuleEvaluationFactoryPtr,
      std::unique_ptr<IDecomposableRuleEvaluationFactory> pruningRuleEvaluationFactoryPtr, uint32 numThreads)
        : lossFactoryPtr_(std::move(lossFactoryPtr)),
          evaluationMeasureFactoryPtr_(std::move(evaluationMeasureFactoryPtr)),
          defaultRuleEvaluationFactoryPtr_(std::move(defaultRuleEvaluationFactoryPtr)),
          regularRuleEvaluationFactoryPtr_(std::move(regularRuleEvaluationFactoryPtr)),
          pruningRuleEvaluationFactoryPtr_(std::move(pruningRuleEvaluationFactoryPtr)), numThreads_(numThreads) {}

    std::unique_ptr<IStatisticsProvider> DenseDecomposableClassificationStatisticsProviderFactory::create(
      const CContiguousView<const uint8>& labelMatrix) const {
        std::unique_ptr<IDecomposableClassificationLoss> lossPtr =
          lossFactoryPtr_->createDecomposableClassificationLoss();
        std::unique_ptr<IClassificationEvaluationMeasure> evaluationMeasurePtr =
          evaluationMeasureFactoryPtr_->createClassificationEvaluationMeasure();
        std::unique_ptr<IDecomposableStatistics<IDecomposableRuleEvaluationFactory>> statisticsPtr =
          createStatistics(std::move(lossPtr), std::move(evaluationMeasurePtr), *defaultRuleEvaluationFactoryPtr_,
                           numThreads_, labelMatrix);
        return std::make_unique<DecomposableStatisticsProvider<IDecomposableRuleEvaluationFactory>>(
          *regularRuleEvaluationFactoryPtr_, *pruningRuleEvaluationFactoryPtr_, std::move(statisticsPtr));
    }

    std::unique_ptr<IStatisticsProvider> DenseDecomposableClassificationStatisticsProviderFactory::create(
      const BinaryCsrView& labelMatrix) const {
        std::unique_ptr<IDecomposableClassificationLoss> lossPtr =
          lossFactoryPtr_->createDecomposableClassificationLoss();
        std::unique_ptr<IClassificationEvaluationMeasure> evaluationMeasurePtr =
          evaluationMeasureFactoryPtr_->createClassificationEvaluationMeasure();
        std::unique_ptr<IDecomposableStatistics<IDecomposableRuleEvaluationFactory>> statisticsPtr =
          createStatistics(std::move(lossPtr), std::move(evaluationMeasurePtr), *defaultRuleEvaluationFactoryPtr_,
                           numThreads_, labelMatrix);
        return std::make_unique<DecomposableStatisticsProvider<IDecomposableRuleEvaluationFactory>>(
          *regularRuleEvaluationFactoryPtr_, *pruningRuleEvaluationFactoryPtr_, std::move(statisticsPtr));
    }

    DenseDecomposableRegressionStatisticsProviderFactory::DenseDecomposableRegressionStatisticsProviderFactory(
      std::unique_ptr<IDecomposableRegressionLossFactory> lossFactoryPtr,
      std::unique_ptr<IRegressionEvaluationMeasureFactory> evaluationMeasureFactoryPtr,
      std::unique_ptr<IDecomposableRuleEvaluationFactory> defaultRuleEvaluationFactoryPtr,
      std::unique_ptr<IDecomposableRuleEvaluationFactory> regularRuleEvaluationFactoryPtr,
      std::unique_ptr<IDecomposableRuleEvaluationFactory> pruningRuleEvaluationFactoryPtr, uint32 numThreads)
        : lossFactoryPtr_(std::move(lossFactoryPtr)),
          evaluationMeasureFactoryPtr_(std::move(evaluationMeasureFactoryPtr)),
          defaultRuleEvaluationFactoryPtr_(std::move(defaultRuleEvaluationFactoryPtr)),
          regularRuleEvaluationFactoryPtr_(std::move(regularRuleEvaluationFactoryPtr)),
          pruningRuleEvaluationFactoryPtr_(std::move(pruningRuleEvaluationFactoryPtr)), numThreads_(numThreads) {}

    std::unique_ptr<IStatisticsProvider> DenseDecomposableRegressionStatisticsProviderFactory::create(
      const CContiguousView<const float32>& regressionMatrix) const {
        std::unique_ptr<IDecomposableRegressionLoss> lossPtr = lossFactoryPtr_->createDecomposableRegressionLoss();
        std::unique_ptr<IRegressionEvaluationMeasure> evaluationMeasurePtr =
          evaluationMeasureFactoryPtr_->createRegressionEvaluationMeasure();
        std::unique_ptr<IDecomposableStatistics<IDecomposableRuleEvaluationFactory>> statisticsPtr =
          createStatistics(std::move(lossPtr), std::move(evaluationMeasurePtr), *defaultRuleEvaluationFactoryPtr_,
                           numThreads_, regressionMatrix);
        return std::make_unique<DecomposableStatisticsProvider<IDecomposableRuleEvaluationFactory>>(
          *regularRuleEvaluationFactoryPtr_, *pruningRuleEvaluationFactoryPtr_, std::move(statisticsPtr));
    }

    std::unique_ptr<IStatisticsProvider> DenseDecomposableRegressionStatisticsProviderFactory::create(
      const CsrView<const float32>& regressionMatrix) const {
        std::unique_ptr<IDecomposableRegressionLoss> lossPtr = lossFactoryPtr_->createDecomposableRegressionLoss();
        std::unique_ptr<IRegressionEvaluationMeasure> evaluationMeasurePtr =
          evaluationMeasureFactoryPtr_->createRegressionEvaluationMeasure();
        std::unique_ptr<IDecomposableStatistics<IDecomposableRuleEvaluationFactory>> statisticsPtr =
          createStatistics(std::move(lossPtr), std::move(evaluationMeasurePtr), *defaultRuleEvaluationFactoryPtr_,
                           numThreads_, regressionMatrix);
        return std::make_unique<DecomposableStatisticsProvider<IDecomposableRuleEvaluationFactory>>(
          *regularRuleEvaluationFactoryPtr_, *pruningRuleEvaluationFactoryPtr_, std::move(statisticsPtr));
    }

}

#ifdef _WIN32
    #pragma warning(pop)
#endif
