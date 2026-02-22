#include "mlrl/boosting/statistics/statistics_provider_decomposable_dense.hpp"

#include "mlrl/common/math/vector_math.hpp"
#include "mlrl/common/util/openmp.hpp"
#include "mlrl/common/util/xsimd.hpp"
#include "statistics_decomposable_dense.hpp"
#include "statistics_provider_decomposable.hpp"

#include <type_traits>

namespace boosting {

    template<typename Loss, typename OutputMatrix, typename EvaluationMeasure, typename VectorMath>
    static inline std::unique_ptr<IDecomposableStatistics<IDecomposableRuleEvaluationFactory>> createStatistics(
      std::unique_ptr<Loss> lossPtr, std::unique_ptr<EvaluationMeasure> evaluationMeasurePtr,
      const IDecomposableRuleEvaluationFactory& ruleEvaluationFactory, MultiThreadingSettings multiThreadingSettings,
      const OutputMatrix& outputMatrix, std::type_identity<VectorMath> vectorMath) {
        using statistic_type = Loss::statistic_type;
        uint32 numExamples = outputMatrix.numRows;
        uint32 numOutputs = outputMatrix.numCols;
        std::unique_ptr<DenseDecomposableStatisticMatrix<statistic_type, VectorMath>> statisticMatrixPtr =
          std::make_unique<DenseDecomposableStatisticMatrix<statistic_type, VectorMath>>(numExamples, numOutputs);
        std::unique_ptr<NumericCContiguousMatrix<statistic_type>> scoreMatrixPtr =
          std::make_unique<NumericCContiguousMatrix<statistic_type>>(numExamples, numOutputs, true);
        const Loss* lossRawPtr = lossPtr.get();
        const OutputMatrix* outputMatrixPtr = &outputMatrix;
        const CContiguousView<statistic_type>* scoreMatrixRawPtr = &scoreMatrixPtr->getView();
        DenseDecomposableStatisticView<statistic_type>* statisticMatrixRawPtr = &statisticMatrixPtr->getView();

#if MULTI_THREADING_SUPPORT_ENABLED
    #pragma omp parallel for firstprivate(numExamples) firstprivate(lossRawPtr) firstprivate(outputMatrixPtr) \
      firstprivate(scoreMatrixRawPtr) firstprivate(statisticMatrixRawPtr) schedule(dynamic) \
      num_threads(multiThreadingSettings.numThreads)
#endif
        for (int64 i = 0; i < numExamples; i++) {
            lossRawPtr->updateDecomposableStatistics(i, *outputMatrixPtr, *scoreMatrixRawPtr, IndexIterator(),
                                                     IndexIterator(outputMatrixPtr->numCols), *statisticMatrixRawPtr);
        }

        return std::make_unique<DenseDecomposableStatistics<Loss, OutputMatrix, EvaluationMeasure, VectorMath>>(
          std::move(lossPtr), std::move(evaluationMeasurePtr), ruleEvaluationFactory, outputMatrix,
          std::move(statisticMatrixPtr), std::move(scoreMatrixPtr));
    }

    template<typename StatisticType, typename VectorMath>
    DenseDecomposableClassificationStatisticsProviderFactory<StatisticType, VectorMath>::
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

    template<typename StatisticType, typename VectorMath>
    std::unique_ptr<IStatisticsProvider>
      DenseDecomposableClassificationStatisticsProviderFactory<StatisticType, VectorMath>::create(
        const CContiguousView<const uint8>& labelMatrix) const {
        std::unique_ptr<IDecomposableClassificationLoss<StatisticType>> lossPtr =
          lossFactoryPtr_->createDecomposableClassificationLoss();
        std::unique_ptr<IClassificationEvaluationMeasure<StatisticType>> evaluationMeasurePtr =
          evaluationMeasureFactoryPtr_->createClassificationEvaluationMeasure();
        std::unique_ptr<IDecomposableStatistics<IDecomposableRuleEvaluationFactory>> statisticsPtr =
          createStatistics(std::move(lossPtr), std::move(evaluationMeasurePtr), *defaultRuleEvaluationFactoryPtr_,
                           multiThreadingSettings_, labelMatrix, std::type_identity<VectorMath> {});
        return std::make_unique<DecomposableStatisticsProvider<IDecomposableRuleEvaluationFactory>>(
          *regularRuleEvaluationFactoryPtr_, *pruningRuleEvaluationFactoryPtr_, std::move(statisticsPtr));
    }

    template<typename StatisticType, typename VectorMath>
    std::unique_ptr<IStatisticsProvider>
      DenseDecomposableClassificationStatisticsProviderFactory<StatisticType, VectorMath>::create(
        const BinaryCsrView& labelMatrix) const {
        std::unique_ptr<IDecomposableClassificationLoss<StatisticType>> lossPtr =
          lossFactoryPtr_->createDecomposableClassificationLoss();
        std::unique_ptr<IClassificationEvaluationMeasure<StatisticType>> evaluationMeasurePtr =
          evaluationMeasureFactoryPtr_->createClassificationEvaluationMeasure();
        std::unique_ptr<IDecomposableStatistics<IDecomposableRuleEvaluationFactory>> statisticsPtr =
          createStatistics(std::move(lossPtr), std::move(evaluationMeasurePtr), *defaultRuleEvaluationFactoryPtr_,
                           multiThreadingSettings_, labelMatrix, std::type_identity<VectorMath> {});
        return std::make_unique<DecomposableStatisticsProvider<IDecomposableRuleEvaluationFactory>>(
          *regularRuleEvaluationFactoryPtr_, *pruningRuleEvaluationFactoryPtr_, std::move(statisticsPtr));
    }

    template class DenseDecomposableClassificationStatisticsProviderFactory<float32, SequentialVectorMath>;
    template class DenseDecomposableClassificationStatisticsProviderFactory<float64, SequentialVectorMath>;

#if SIMD_SUPPORT_ENABLED
    template class DenseDecomposableClassificationStatisticsProviderFactory<float32, SimdVectorMath>;
    template class DenseDecomposableClassificationStatisticsProviderFactory<float64, SimdVectorMath>;
#endif

    template<typename StatisticType, typename VectorMath>
    DenseDecomposableRegressionStatisticsProviderFactory<StatisticType, VectorMath>::
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

    template<typename StatisticType, typename VectorMath>
    std::unique_ptr<IStatisticsProvider>
      DenseDecomposableRegressionStatisticsProviderFactory<StatisticType, VectorMath>::create(
        const CContiguousView<const float32>& regressionMatrix) const {
        std::unique_ptr<IDecomposableRegressionLoss<StatisticType>> lossPtr =
          lossFactoryPtr_->createDecomposableRegressionLoss();
        std::unique_ptr<IRegressionEvaluationMeasure<StatisticType>> evaluationMeasurePtr =
          evaluationMeasureFactoryPtr_->createRegressionEvaluationMeasure();
        std::unique_ptr<IDecomposableStatistics<IDecomposableRuleEvaluationFactory>> statisticsPtr =
          createStatistics(std::move(lossPtr), std::move(evaluationMeasurePtr), *defaultRuleEvaluationFactoryPtr_,
                           multiThreadingSettings_, regressionMatrix, std::type_identity<VectorMath> {});
        return std::make_unique<DecomposableStatisticsProvider<IDecomposableRuleEvaluationFactory>>(
          *regularRuleEvaluationFactoryPtr_, *pruningRuleEvaluationFactoryPtr_, std::move(statisticsPtr));
    }

    template<typename StatisticType, typename VectorMath>
    std::unique_ptr<IStatisticsProvider>
      DenseDecomposableRegressionStatisticsProviderFactory<StatisticType, VectorMath>::create(
        const CsrView<const float32>& regressionMatrix) const {
        std::unique_ptr<IDecomposableRegressionLoss<StatisticType>> lossPtr =
          lossFactoryPtr_->createDecomposableRegressionLoss();
        std::unique_ptr<IRegressionEvaluationMeasure<StatisticType>> evaluationMeasurePtr =
          evaluationMeasureFactoryPtr_->createRegressionEvaluationMeasure();
        std::unique_ptr<IDecomposableStatistics<IDecomposableRuleEvaluationFactory>> statisticsPtr =
          createStatistics(std::move(lossPtr), std::move(evaluationMeasurePtr), *defaultRuleEvaluationFactoryPtr_,
                           multiThreadingSettings_, regressionMatrix, std::type_identity<VectorMath> {});
        return std::make_unique<DecomposableStatisticsProvider<IDecomposableRuleEvaluationFactory>>(
          *regularRuleEvaluationFactoryPtr_, *pruningRuleEvaluationFactoryPtr_, std::move(statisticsPtr));
    }

    template class DenseDecomposableRegressionStatisticsProviderFactory<float32, SequentialVectorMath>;
    template class DenseDecomposableRegressionStatisticsProviderFactory<float64, SequentialVectorMath>;

#if SIMD_SUPPORT_ENABLED
    template class DenseDecomposableRegressionStatisticsProviderFactory<float32, SimdVectorMath>;
    template class DenseDecomposableRegressionStatisticsProviderFactory<float64, SimdVectorMath>;
#endif

}
