#include "mlrl/boosting/statistics/statistics_provider_decomposable_dense.hpp"

#include "mlrl/common/util/openmp.hpp"
#include "statistics_decomposable_dense.hpp"
#include "statistics_provider_decomposable.hpp"

namespace boosting {

    template<typename Loss, typename OutputMatrix, typename EvaluationMeasure>
    static inline std::unique_ptr<IDecomposableStatistics<IDecomposableRuleEvaluationFactory>> createStatistics(
      std::unique_ptr<Loss> lossPtr, std::unique_ptr<EvaluationMeasure> evaluationMeasurePtr,
      const IDecomposableRuleEvaluationFactory& ruleEvaluationFactory, MultiThreadingSettings multiThreadingSettings,
      const OutputMatrix& outputMatrix) {
        typedef typename Loss::statistic_type statistic_type;
        uint32 numExamples = outputMatrix.numRows;
        uint32 numOutputs = outputMatrix.numCols;
        std::unique_ptr<DenseDecomposableStatisticMatrix<statistic_type>> statisticMatrixPtr =
          std::make_unique<DenseDecomposableStatisticMatrix<statistic_type>>(numExamples, numOutputs);
        std::unique_ptr<NumericCContiguousMatrix<statistic_type>> scoreMatrixPtr =
          std::make_unique<NumericCContiguousMatrix<statistic_type>>(numExamples, numOutputs, true);
        const Loss* lossRawPtr = lossPtr.get();
        const OutputMatrix* outputMatrixPtr = &outputMatrix;
        const CContiguousView<statistic_type>* scoreMatrixRawPtr = &scoreMatrixPtr->getView();
        CContiguousView<Statistic<statistic_type>>* statisticMatrixRawPtr = &statisticMatrixPtr->getView();

#if MULTI_THREADING_SUPPORT_ENABLED
    #pragma omp parallel for firstprivate(numExamples) firstprivate(lossRawPtr) firstprivate(outputMatrixPtr) \
      firstprivate(scoreMatrixRawPtr) firstprivate(statisticMatrixRawPtr) schedule(dynamic) \
      num_threads(multiThreadingSettings.numThreads)
#endif
        for (int64 i = 0; i < numExamples; i++) {
            lossRawPtr->updateDecomposableStatistics(i, *outputMatrixPtr, *scoreMatrixRawPtr, IndexIterator(),
                                                     IndexIterator(outputMatrixPtr->numCols), *statisticMatrixRawPtr);
        }

        return std::make_unique<DenseDecomposableStatistics<Loss, OutputMatrix, EvaluationMeasure>>(
          std::move(lossPtr), std::move(evaluationMeasurePtr), ruleEvaluationFactory, outputMatrix,
          std::move(statisticMatrixPtr), std::move(scoreMatrixPtr));
    }

    template<typename StatisticType>
    DenseDecomposableClassificationStatisticsProviderFactory<StatisticType>::
      DenseDecomposableClassificationStatisticsProviderFactory(
        std::unique_ptr<IDecomposableClassificationLossFactory<StatisticType>> lossFactoryPtr,
        std::unique_ptr<IClassificationEvaluationMeasureFactory<StatisticType>> evaluationMeasureFactoryPtr,
        std::unique_ptr<IDecomposableRuleEvaluationFactory> defaultRuleEvaluationFactoryPtr,
        std::unique_ptr<IDecomposableRuleEvaluationFactory> regularRuleEvaluationFactoryPtr,
        std::unique_ptr<IDecomposableRuleEvaluationFactory> pruningRuleEvaluationFactoryPtr,
        MultiThreadingSettings multiThreadingSettings)
        : lossFactoryPtr_(std::move(lossFactoryPtr)),
          evaluationMeasureFactoryPtr_(std::move(evaluationMeasureFactoryPtr)),
          defaultRuleEvaluationFactoryPtr_(std::move(defaultRuleEvaluationFactoryPtr)),
          regularRuleEvaluationFactoryPtr_(std::move(regularRuleEvaluationFactoryPtr)),
          pruningRuleEvaluationFactoryPtr_(std::move(pruningRuleEvaluationFactoryPtr)),
          multiThreadingSettings_(multiThreadingSettings) {}

    template<typename StatisticType>
    std::unique_ptr<IStatisticsProvider>
      DenseDecomposableClassificationStatisticsProviderFactory<StatisticType>::create(
        const CContiguousView<const uint8>& labelMatrix) const {
        std::unique_ptr<IDecomposableClassificationLoss<StatisticType>> lossPtr =
          lossFactoryPtr_->createDecomposableClassificationLoss();
        std::unique_ptr<IClassificationEvaluationMeasure<StatisticType>> evaluationMeasurePtr =
          evaluationMeasureFactoryPtr_->createClassificationEvaluationMeasure();
        std::unique_ptr<IDecomposableStatistics<IDecomposableRuleEvaluationFactory>> statisticsPtr =
          createStatistics(std::move(lossPtr), std::move(evaluationMeasurePtr), *defaultRuleEvaluationFactoryPtr_,
                           multiThreadingSettings_, labelMatrix);
        return std::make_unique<DecomposableStatisticsProvider<IDecomposableRuleEvaluationFactory>>(
          *regularRuleEvaluationFactoryPtr_, *pruningRuleEvaluationFactoryPtr_, std::move(statisticsPtr));
    }

    template<typename StatisticType>
    std::unique_ptr<IStatisticsProvider>
      DenseDecomposableClassificationStatisticsProviderFactory<StatisticType>::create(
        const BinaryCsrView& labelMatrix) const {
        std::unique_ptr<IDecomposableClassificationLoss<StatisticType>> lossPtr =
          lossFactoryPtr_->createDecomposableClassificationLoss();
        std::unique_ptr<IClassificationEvaluationMeasure<StatisticType>> evaluationMeasurePtr =
          evaluationMeasureFactoryPtr_->createClassificationEvaluationMeasure();
        std::unique_ptr<IDecomposableStatistics<IDecomposableRuleEvaluationFactory>> statisticsPtr =
          createStatistics(std::move(lossPtr), std::move(evaluationMeasurePtr), *defaultRuleEvaluationFactoryPtr_,
                           multiThreadingSettings_, labelMatrix);
        return std::make_unique<DecomposableStatisticsProvider<IDecomposableRuleEvaluationFactory>>(
          *regularRuleEvaluationFactoryPtr_, *pruningRuleEvaluationFactoryPtr_, std::move(statisticsPtr));
    }

    template class DenseDecomposableClassificationStatisticsProviderFactory<float32>;
    template class DenseDecomposableClassificationStatisticsProviderFactory<float64>;

    template<typename StatisticType>
    DenseDecomposableRegressionStatisticsProviderFactory<StatisticType>::
      DenseDecomposableRegressionStatisticsProviderFactory(
        std::unique_ptr<IDecomposableRegressionLossFactory<StatisticType>> lossFactoryPtr,
        std::unique_ptr<IRegressionEvaluationMeasureFactory<StatisticType>> evaluationMeasureFactoryPtr,
        std::unique_ptr<IDecomposableRuleEvaluationFactory> defaultRuleEvaluationFactoryPtr,
        std::unique_ptr<IDecomposableRuleEvaluationFactory> regularRuleEvaluationFactoryPtr,
        std::unique_ptr<IDecomposableRuleEvaluationFactory> pruningRuleEvaluationFactoryPtr,
        MultiThreadingSettings multiThreadingSettings)
        : lossFactoryPtr_(std::move(lossFactoryPtr)),
          evaluationMeasureFactoryPtr_(std::move(evaluationMeasureFactoryPtr)),
          defaultRuleEvaluationFactoryPtr_(std::move(defaultRuleEvaluationFactoryPtr)),
          regularRuleEvaluationFactoryPtr_(std::move(regularRuleEvaluationFactoryPtr)),
          pruningRuleEvaluationFactoryPtr_(std::move(pruningRuleEvaluationFactoryPtr)),
          multiThreadingSettings_(multiThreadingSettings) {}

    template<typename StatisticType>
    std::unique_ptr<IStatisticsProvider> DenseDecomposableRegressionStatisticsProviderFactory<StatisticType>::create(
      const CContiguousView<const float32>& regressionMatrix) const {
        std::unique_ptr<IDecomposableRegressionLoss<StatisticType>> lossPtr =
          lossFactoryPtr_->createDecomposableRegressionLoss();
        std::unique_ptr<IRegressionEvaluationMeasure<StatisticType>> evaluationMeasurePtr =
          evaluationMeasureFactoryPtr_->createRegressionEvaluationMeasure();
        std::unique_ptr<IDecomposableStatistics<IDecomposableRuleEvaluationFactory>> statisticsPtr =
          createStatistics(std::move(lossPtr), std::move(evaluationMeasurePtr), *defaultRuleEvaluationFactoryPtr_,
                           multiThreadingSettings_, regressionMatrix);
        return std::make_unique<DecomposableStatisticsProvider<IDecomposableRuleEvaluationFactory>>(
          *regularRuleEvaluationFactoryPtr_, *pruningRuleEvaluationFactoryPtr_, std::move(statisticsPtr));
    }

    template<typename StatisticType>
    std::unique_ptr<IStatisticsProvider> DenseDecomposableRegressionStatisticsProviderFactory<StatisticType>::create(
      const CsrView<const float32>& regressionMatrix) const {
        std::unique_ptr<IDecomposableRegressionLoss<StatisticType>> lossPtr =
          lossFactoryPtr_->createDecomposableRegressionLoss();
        std::unique_ptr<IRegressionEvaluationMeasure<StatisticType>> evaluationMeasurePtr =
          evaluationMeasureFactoryPtr_->createRegressionEvaluationMeasure();
        std::unique_ptr<IDecomposableStatistics<IDecomposableRuleEvaluationFactory>> statisticsPtr =
          createStatistics(std::move(lossPtr), std::move(evaluationMeasurePtr), *defaultRuleEvaluationFactoryPtr_,
                           multiThreadingSettings_, regressionMatrix);
        return std::make_unique<DecomposableStatisticsProvider<IDecomposableRuleEvaluationFactory>>(
          *regularRuleEvaluationFactoryPtr_, *pruningRuleEvaluationFactoryPtr_, std::move(statisticsPtr));
    }

    template class DenseDecomposableRegressionStatisticsProviderFactory<float32>;
    template class DenseDecomposableRegressionStatisticsProviderFactory<float64>;

}
