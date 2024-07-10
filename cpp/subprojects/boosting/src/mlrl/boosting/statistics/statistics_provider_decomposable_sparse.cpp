#ifdef _WIN32
    #pragma warning(push)
    #pragma warning(disable : 4250)
#endif

#include "mlrl/boosting/statistics/statistics_provider_decomposable_sparse.hpp"

#include "mlrl/boosting/data/matrix_sparse_set_numeric.hpp"
#include "mlrl/boosting/data/vector_statistic_decomposable_sparse.hpp"
#include "mlrl/common/util/openmp.hpp"
#include "statistics_decomposable_common.hpp"
#include "statistics_provider_decomposable.hpp"

namespace boosting {

    /**
     * A matrix that stores gradients and Hessians that have been calculated using a decomposable loss function in the
     * list of lists (LIL) format.
     */
    class SparseDecomposableStatisticMatrix final : public MatrixDecorator<SparseSetView<Tuple<float64>>> {
        public:

            /**
             * @param numRows   The number of rows in the matrix
             * @param numCols   The number of columns in the matrix
             */
            SparseDecomposableStatisticMatrix(uint32 numRows, uint32 numCols)
                : MatrixDecorator<SparseSetView<Tuple<float64>>>(SparseSetView<Tuple<float64>>(numRows, numCols)) {}
    };

    /**
     * Provides access to gradients and Hessians that have been calculated according to a decomposable loss function
     * and are stored using sparse data structures.
     *
     * @tparam OutputMatrix The type of the matrix that provides access to the ground truth of the training examples
     */
    template<typename OutputMatrix>
    class SparseDecomposableStatistics final
        : public AbstractDecomposableStatistics<OutputMatrix, SparseDecomposableStatisticVector,
                                                SparseDecomposableStatisticMatrix, NumericSparseSetMatrix<float64>,
                                                ISparseDecomposableLoss, ISparseEvaluationMeasure,
                                                ISparseDecomposableRuleEvaluationFactory> {
        public:

            /**
             * @param lossPtr               An unique pointer to an object of template type `LossFunction` that
             *                              implements the loss function that should be used for calculating gradients
             *                              and Hessians
             * @param evaluationMeasurePtr  An unique pointer to an object of type `ISparseEvaluationMeasure` that
             *                              implements the evaluation measure that should be used to assess the quality
             *                              of predictions for a specific statistic
             * @param ruleEvaluationFactory A reference to an object of type `ISparseDecomposableRuleEvaluationFactory`,
             *                              that allows to create instances of the class that is used for calculating
             *                              the predictions of rules, as well as their overall quality
             * @param outputMatrix          A reference to an object of template type `OutputMatrix` that provides
             *                              access to the outputs of the training examples
             * @param statisticViewPtr      An unique pointer to an object of type `SparseDecomposableStatisticMatrix`
             *                              that provides access to the gradients and Hessians
             * @param scoreMatrixPtr        An unique pointer to an object of type `NumericSparseSetMatrix` that stores
             *                              the currently predicted scores
             */
            SparseDecomposableStatistics(std::unique_ptr<ISparseDecomposableLoss> lossPtr,
                                         std::unique_ptr<ISparseEvaluationMeasure> evaluationMeasurePtr,
                                         const ISparseDecomposableRuleEvaluationFactory& ruleEvaluationFactory,
                                         const OutputMatrix& outputMatrix,
                                         std::unique_ptr<SparseDecomposableStatisticMatrix> statisticViewPtr,
                                         std::unique_ptr<NumericSparseSetMatrix<float64>> scoreMatrixPtr)
                : AbstractDecomposableStatistics<OutputMatrix, SparseDecomposableStatisticVector,
                                                 SparseDecomposableStatisticMatrix, NumericSparseSetMatrix<float64>,
                                                 ISparseDecomposableLoss, ISparseEvaluationMeasure,
                                                 ISparseDecomposableRuleEvaluationFactory>(
                    std::move(lossPtr), std::move(evaluationMeasurePtr), ruleEvaluationFactory, outputMatrix,
                    std::move(statisticViewPtr), std::move(scoreMatrixPtr)) {}

            /**
             * @see `IBoostingStatistics::visitScoreMatrix`
             */
            void visitScoreMatrix(IBoostingStatistics::DenseScoreMatrixVisitor denseVisitor,
                                  IBoostingStatistics::SparseScoreMatrixVisitor sparseVisitor) const override {
                sparseVisitor(this->scoreMatrixPtr_->getView());
            }
    };

    template<typename OutputMatrix>
    static inline std::unique_ptr<IDecomposableStatistics<ISparseDecomposableRuleEvaluationFactory>> createStatistics(
      const ISparseDecomposableLossFactory& lossFactory,
      const ISparseEvaluationMeasureFactory& evaluationMeasureFactory,
      const ISparseDecomposableRuleEvaluationFactory& ruleEvaluationFactory, uint32 numThreads,
      const OutputMatrix& outputMatrix) {
        uint32 numExamples = outputMatrix.numRows;
        uint32 numOutputs = outputMatrix.numCols;
        std::unique_ptr<ISparseDecomposableLoss> lossPtr = lossFactory.createSparseDecomposableLoss();
        std::unique_ptr<ISparseEvaluationMeasure> evaluationMeasurePtr =
          evaluationMeasureFactory.createSparseEvaluationMeasure();
        std::unique_ptr<SparseDecomposableStatisticMatrix> statisticMatrixPtr =
          std::make_unique<SparseDecomposableStatisticMatrix>(numExamples, numOutputs);
        std::unique_ptr<NumericSparseSetMatrix<float64>> scoreMatrixPtr =
          std::make_unique<NumericSparseSetMatrix<float64>>(numExamples, numOutputs);
        const ISparseDecomposableLoss* lossRawPtr = lossPtr.get();
        const OutputMatrix* outputMatrixPtr = &outputMatrix;
        const SparseSetView<float64>* scoreMatrixRawPtr = &scoreMatrixPtr->getView();
        SparseSetView<Tuple<float64>>* statisticMatrixRawPtr = &statisticMatrixPtr->getView();

#if MULTI_THREADING_SUPPORT_ENABLED
    #pragma omp parallel for firstprivate(numExamples) firstprivate(lossRawPtr) firstprivate(outputMatrixPtr) \
      firstprivate(scoreMatrixRawPtr) firstprivate(statisticMatrixRawPtr) schedule(dynamic) num_threads(numThreads)
#endif
        for (int64 i = 0; i < numExamples; i++) {
            lossRawPtr->updateDecomposableStatistics(i, *outputMatrixPtr, *scoreMatrixRawPtr, IndexIterator(),
                                                     IndexIterator(outputMatrixPtr->numCols), *statisticMatrixRawPtr);
        }

        return std::make_unique<SparseDecomposableStatistics<OutputMatrix>>(
          std::move(lossPtr), std::move(evaluationMeasurePtr), ruleEvaluationFactory, outputMatrix,
          std::move(statisticMatrixPtr), std::move(scoreMatrixPtr));
    }

    SparseDecomposableStatisticsProviderFactory::SparseDecomposableStatisticsProviderFactory(
      std::unique_ptr<ISparseDecomposableLossFactory> lossFactoryPtr,
      std::unique_ptr<ISparseEvaluationMeasureFactory> evaluationMeasureFactoryPtr,
      std::unique_ptr<ISparseDecomposableRuleEvaluationFactory> regularRuleEvaluationFactoryPtr,
      std::unique_ptr<ISparseDecomposableRuleEvaluationFactory> pruningRuleEvaluationFactoryPtr, uint32 numThreads)
        : lossFactoryPtr_(std::move(lossFactoryPtr)),
          evaluationMeasureFactoryPtr_(std::move(evaluationMeasureFactoryPtr)),
          regularRuleEvaluationFactoryPtr_(std::move(regularRuleEvaluationFactoryPtr)),
          pruningRuleEvaluationFactoryPtr_(std::move(pruningRuleEvaluationFactoryPtr)), numThreads_(numThreads) {}

    std::unique_ptr<IStatisticsProvider> SparseDecomposableStatisticsProviderFactory::create(
      const CContiguousView<const uint8>& labelMatrix) const {
        std::unique_ptr<IDecomposableStatistics<ISparseDecomposableRuleEvaluationFactory>> statisticsPtr =
          createStatistics(*lossFactoryPtr_, *evaluationMeasureFactoryPtr_, *regularRuleEvaluationFactoryPtr_,
                           numThreads_, labelMatrix);
        return std::make_unique<DecomposableStatisticsProvider<ISparseDecomposableRuleEvaluationFactory>>(
          *regularRuleEvaluationFactoryPtr_, *pruningRuleEvaluationFactoryPtr_, std::move(statisticsPtr));
    }

    std::unique_ptr<IStatisticsProvider> SparseDecomposableStatisticsProviderFactory::create(
      const BinaryCsrView& labelMatrix) const {
        std::unique_ptr<IDecomposableStatistics<ISparseDecomposableRuleEvaluationFactory>> statisticsPtr =
          createStatistics(*lossFactoryPtr_, *evaluationMeasureFactoryPtr_, *regularRuleEvaluationFactoryPtr_,
                           numThreads_, labelMatrix);
        return std::make_unique<DecomposableStatisticsProvider<ISparseDecomposableRuleEvaluationFactory>>(
          *regularRuleEvaluationFactoryPtr_, *pruningRuleEvaluationFactoryPtr_, std::move(statisticsPtr));
    }

    std::unique_ptr<IStatisticsProvider> SparseDecomposableStatisticsProviderFactory::create(
      const CContiguousView<const float32>& regressionMatrix) const {
        // TODO
        return nullptr;
    }

    std::unique_ptr<IStatisticsProvider> SparseDecomposableStatisticsProviderFactory::create(
      const CsrView<const float32>& regressionMatrix) const {
        // TODO
        return nullptr;
    }
}

#ifdef _WIN32
    #pragma warning(pop)
#endif
