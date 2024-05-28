#ifdef _WIN32
    #pragma warning(push)
    #pragma warning(disable : 4250)
#endif

#include "mlrl/boosting/statistics/statistics_provider_decomposable_dense.hpp"

#include "mlrl/common/util/openmp.hpp"
#include "statistics_decomposable_dense.hpp"
#include "statistics_provider_decomposable.hpp"

namespace boosting {

    template<typename LabelMatrix>
    static inline std::unique_ptr<IDecomposableStatistics<IDecomposableRuleEvaluationFactory>> createStatistics(
      const ILabelWiseLossFactory& lossFactory, const IEvaluationMeasureFactory& evaluationMeasureFactory,
      const IDecomposableRuleEvaluationFactory& ruleEvaluationFactory, uint32 numThreads,
      const LabelMatrix& labelMatrix) {
        uint32 numExamples = labelMatrix.numRows;
        uint32 numLabels = labelMatrix.numCols;
        std::unique_ptr<ILabelWiseLoss> lossPtr = lossFactory.createLabelWiseLoss();
        std::unique_ptr<IEvaluationMeasure> evaluationMeasurePtr = evaluationMeasureFactory.createEvaluationMeasure();
        std::unique_ptr<DenseDecomposableStatisticMatrix> statisticMatrixPtr =
          std::make_unique<DenseDecomposableStatisticMatrix>(numExamples, numLabels);
        std::unique_ptr<NumericCContiguousMatrix<float64>> scoreMatrixPtr =
          std::make_unique<NumericCContiguousMatrix<float64>>(numExamples, numLabels, true);
        const ILabelWiseLoss* lossRawPtr = lossPtr.get();
        const LabelMatrix* labelMatrixPtr = &labelMatrix;
        const CContiguousView<float64>* scoreMatrixRawPtr = &scoreMatrixPtr->getView();
        CContiguousView<Tuple<float64>>* statisticMatrixRawPtr = &statisticMatrixPtr->getView();

#if MULTI_THREADING_SUPPORT_ENABLED
    #pragma omp parallel for firstprivate(numExamples) firstprivate(lossRawPtr) firstprivate(labelMatrixPtr) \
      firstprivate(scoreMatrixRawPtr) firstprivate(statisticMatrixRawPtr) schedule(dynamic) num_threads(numThreads)
#endif
        for (int64 i = 0; i < numExamples; i++) {
            lossRawPtr->updateDecomposableStatistics(i, *labelMatrixPtr, *scoreMatrixRawPtr, IndexIterator(),
                                                     IndexIterator(labelMatrixPtr->numCols), *statisticMatrixRawPtr);
        }

        return std::make_unique<DenseDecomposableStatistics<LabelMatrix>>(
          std::move(lossPtr), std::move(evaluationMeasurePtr), ruleEvaluationFactory, labelMatrix,
          std::move(statisticMatrixPtr), std::move(scoreMatrixPtr));
    }

    DenseDecomposableStatisticsProviderFactory::DenseDecomposableStatisticsProviderFactory(
      std::unique_ptr<ILabelWiseLossFactory> lossFactoryPtr,
      std::unique_ptr<IEvaluationMeasureFactory> evaluationMeasureFactoryPtr,
      std::unique_ptr<IDecomposableRuleEvaluationFactory> defaultRuleEvaluationFactoryPtr,
      std::unique_ptr<IDecomposableRuleEvaluationFactory> regularRuleEvaluationFactoryPtr,
      std::unique_ptr<IDecomposableRuleEvaluationFactory> pruningRuleEvaluationFactoryPtr, uint32 numThreads)
        : lossFactoryPtr_(std::move(lossFactoryPtr)),
          evaluationMeasureFactoryPtr_(std::move(evaluationMeasureFactoryPtr)),
          defaultRuleEvaluationFactoryPtr_(std::move(defaultRuleEvaluationFactoryPtr)),
          regularRuleEvaluationFactoryPtr_(std::move(regularRuleEvaluationFactoryPtr)),
          pruningRuleEvaluationFactoryPtr_(std::move(pruningRuleEvaluationFactoryPtr)), numThreads_(numThreads) {}

    std::unique_ptr<IStatisticsProvider> DenseDecomposableStatisticsProviderFactory::create(
      const CContiguousView<const uint8>& labelMatrix) const {
        std::unique_ptr<IDecomposableStatistics<IDecomposableRuleEvaluationFactory>> statisticsPtr = createStatistics(
          *lossFactoryPtr_, *evaluationMeasureFactoryPtr_, *defaultRuleEvaluationFactoryPtr_, numThreads_, labelMatrix);
        return std::make_unique<DecomposableStatisticsProvider<IDecomposableRuleEvaluationFactory>>(
          *regularRuleEvaluationFactoryPtr_, *pruningRuleEvaluationFactoryPtr_, std::move(statisticsPtr));
    }

    std::unique_ptr<IStatisticsProvider> DenseDecomposableStatisticsProviderFactory::create(
      const BinaryCsrView& labelMatrix) const {
        std::unique_ptr<IDecomposableStatistics<IDecomposableRuleEvaluationFactory>> statisticsPtr = createStatistics(
          *lossFactoryPtr_, *evaluationMeasureFactoryPtr_, *defaultRuleEvaluationFactoryPtr_, numThreads_, labelMatrix);
        return std::make_unique<DecomposableStatisticsProvider<IDecomposableRuleEvaluationFactory>>(
          *regularRuleEvaluationFactoryPtr_, *pruningRuleEvaluationFactoryPtr_, std::move(statisticsPtr));
    }

}

#ifdef _WIN32
    #pragma warning(pop)
#endif
