#ifdef _WIN32
    #pragma warning(push)
    #pragma warning(disable : 4250)
#endif

#include "mlrl/boosting/statistics/statistics_provider_label_wise_sparse.hpp"

#include "mlrl/boosting/data/matrix_sparse_set_numeric.hpp"
#include "mlrl/boosting/data/statistic_vector_label_wise_sparse.hpp"
#include "mlrl/boosting/data/view_histogram_label_wise_sparse.hpp"
#include "mlrl/common/util/openmp.hpp"
#include "statistics_label_wise_common.hpp"
#include "statistics_provider_label_wise.hpp"

namespace boosting {

    /**
     * A matrix that stores gradients and Hessians that have been calculated using a label-wise decomposable loss
     * function in the list of lists (LIL) format.
     */
    class SparseLabelWiseStatisticMatrix final : public MatrixDecorator<SparseSetView<Tuple<float64>>> {
        public:

            /**
             * @param numRows   The number of rows in the matrix
             * @param numCols   The number of columns in the matrix
             */
            SparseLabelWiseStatisticMatrix(uint32 numRows, uint32 numCols)
                : MatrixDecorator<SparseSetView<Tuple<float64>>>(SparseSetView<Tuple<float64>>(
                  AllocatedListOfLists<IndexedValue<Tuple<float64>>>(numRows, numCols),
                  AllocatedCContiguousView<uint32>(numRows, numCols), numRows, numCols)) {}
    };

    /**
     * A histogram that stores gradients and Hessians that have been calculated using a label-wise decomposable
     * loss function in the list of lists (LIL) format.
     */
    class SparseLabelWiseHistogram final
        : public ClearableViewDecorator<MatrixDecorator<SparseLabelWiseHistogramView>> {
        public:

            /**
             * @param numBins   The number of bins in the histogram
             * @param numCols   The number of columns in the histogram
             */
            SparseLabelWiseHistogram(uint32 numBins, uint32 numCols)
                : ClearableViewDecorator<MatrixDecorator<SparseLabelWiseHistogramView>>(
                  SparseLabelWiseHistogramView(AllocatedCContiguousView<Triple<float64>>(numBins, numCols),
                                               AllocatedVector<float64>(numBins, true), numBins, numCols)) {}

            /**
             * Adds all gradients and Hessians in a vector to a specific row of this histogram. The gradients and
             * Hessians to be added are multiplied by a specific weight.
             *
             * @param row       The row
             * @param begin     An iterator to the beginning of the vector
             * @param end       An iterator to the end of the vector
             * @param weight    The weight, the gradients and Hessians should be multiplied by
             */
            void addToRow(uint32 row, SparseSetView<Tuple<float64>>::value_const_iterator begin,
                          SparseSetView<Tuple<float64>>::value_const_iterator end, float64 weight) {
                if (weight != 0) {
                    this->view.secondView[row] += weight;
                    addToSparseLabelWiseStatisticVector(this->view.firstView.values_begin(row), begin, end, weight);
                }
            }
    };

    /**
     * Provides access to gradients and Hessians that have been calculated according to a differentiable loss
     * function that is applied label-wise and are stored using sparse data structures.
     *
     * @tparam LabelMatrix The type of the matrix that provides access to the labels of the training examples
     */
    template<typename LabelMatrix>
    class SparseLabelWiseStatistics final
        : public AbstractLabelWiseStatistics<LabelMatrix, SparseLabelWiseStatisticVector,
                                             SparseLabelWiseStatisticMatrix, SparseLabelWiseHistogram,
                                             NumericSparseSetMatrix<float64>, ISparseLabelWiseLoss,
                                             ISparseEvaluationMeasure, ISparseLabelWiseRuleEvaluationFactory> {
        public:

            /**
             * @param lossPtr               An unique pointer to an object of template type `LossFunction` that
             *                              implements the loss function that should be used for calculating gradients
             *                              and Hessians
             * @param evaluationMeasurePtr  An unique pointer to an object of type `ISparseEvaluationMeasure` that
             *                              implements the evaluation measure that should be used to assess the quality
             *                              of predictions for a specific statistic
             * @param ruleEvaluationFactory A reference to an object of type `ISparseLabelWiseRuleEvaluationFactory`,
             *                              that allows to create instances of the class that is used for calculating
             *                              the predictions of rules, as well as their overall quality
             * @param labelMatrix           A reference to an object of template type `LabelMatrix` that provides access
             *                              to the labels of the training examples
             * @param statisticViewPtr      An unique pointer to an object of type `SparseLabelWiseStatisticMatrix` that
             *                              provides access to the gradients and Hessians
             * @param scoreMatrixPtr        An unique pointer to an object of type `NumericSparseSetMatrix` that stores
             *                              the currently predicted scores
             */
            SparseLabelWiseStatistics(std::unique_ptr<ISparseLabelWiseLoss> lossPtr,
                                      std::unique_ptr<ISparseEvaluationMeasure> evaluationMeasurePtr,
                                      const ISparseLabelWiseRuleEvaluationFactory& ruleEvaluationFactory,
                                      const LabelMatrix& labelMatrix,
                                      std::unique_ptr<SparseLabelWiseStatisticMatrix> statisticViewPtr,
                                      std::unique_ptr<NumericSparseSetMatrix<float64>> scoreMatrixPtr)
                : AbstractLabelWiseStatistics<LabelMatrix, SparseLabelWiseStatisticVector,
                                              SparseLabelWiseStatisticMatrix, SparseLabelWiseHistogram,
                                              NumericSparseSetMatrix<float64>, ISparseLabelWiseLoss,
                                              ISparseEvaluationMeasure, ISparseLabelWiseRuleEvaluationFactory>(
                  std::move(lossPtr), std::move(evaluationMeasurePtr), ruleEvaluationFactory, labelMatrix,
                  std::move(statisticViewPtr), std::move(scoreMatrixPtr)) {}

            /**
             * @see `IBoostingStatistics::visitScoreMatrix`
             */
            void visitScoreMatrix(IBoostingStatistics::DenseScoreMatrixVisitor denseVisitor,
                                  IBoostingStatistics::SparseScoreMatrixVisitor sparseVisitor) const override {
                sparseVisitor(this->scoreMatrixPtr_->getView());
            }
    };

    template<typename LabelMatrix>
    static inline std::unique_ptr<ILabelWiseStatistics<ISparseLabelWiseRuleEvaluationFactory>> createStatistics(
      const ISparseLabelWiseLossFactory& lossFactory, const ISparseEvaluationMeasureFactory& evaluationMeasureFactory,
      const ISparseLabelWiseRuleEvaluationFactory& ruleEvaluationFactory, uint32 numThreads,
      const LabelMatrix& labelMatrix) {
        uint32 numExamples = labelMatrix.numRows;
        uint32 numLabels = labelMatrix.numCols;
        std::unique_ptr<ISparseLabelWiseLoss> lossPtr = lossFactory.createSparseLabelWiseLoss();
        std::unique_ptr<ISparseEvaluationMeasure> evaluationMeasurePtr =
          evaluationMeasureFactory.createSparseEvaluationMeasure();
        std::unique_ptr<SparseLabelWiseStatisticMatrix> statisticMatrixPtr =
          std::make_unique<SparseLabelWiseStatisticMatrix>(numExamples, numLabels);
        std::unique_ptr<NumericSparseSetMatrix<float64>> scoreMatrixPtr =
          std::make_unique<NumericSparseSetMatrix<float64>>(numExamples, numLabels);
        const ISparseLabelWiseLoss* lossRawPtr = lossPtr.get();
        const LabelMatrix* labelMatrixPtr = &labelMatrix;
        const SparseSetView<float64>* scoreMatrixRawPtr = &scoreMatrixPtr->getView();
        SparseSetView<Tuple<float64>>* statisticMatrixRawPtr = &statisticMatrixPtr->getView();

#if MULTI_THREADING_SUPPORT_ENABLED
    #pragma omp parallel for firstprivate(numExamples) firstprivate(lossRawPtr) firstprivate(labelMatrixPtr) \
      firstprivate(scoreMatrixRawPtr) firstprivate(statisticMatrixRawPtr) schedule(dynamic) num_threads(numThreads)
#endif
        for (int64 i = 0; i < numExamples; i++) {
            lossRawPtr->updateLabelWiseStatistics(i, *labelMatrixPtr, *scoreMatrixRawPtr, IndexIterator(),
                                                  IndexIterator(labelMatrixPtr->numCols), *statisticMatrixRawPtr);
        }

        return std::make_unique<SparseLabelWiseStatistics<LabelMatrix>>(
          std::move(lossPtr), std::move(evaluationMeasurePtr), ruleEvaluationFactory, labelMatrix,
          std::move(statisticMatrixPtr), std::move(scoreMatrixPtr));
    }

    SparseLabelWiseStatisticsProviderFactory::SparseLabelWiseStatisticsProviderFactory(
      std::unique_ptr<ISparseLabelWiseLossFactory> lossFactoryPtr,
      std::unique_ptr<ISparseEvaluationMeasureFactory> evaluationMeasureFactoryPtr,
      std::unique_ptr<ISparseLabelWiseRuleEvaluationFactory> regularRuleEvaluationFactoryPtr,
      std::unique_ptr<ISparseLabelWiseRuleEvaluationFactory> pruningRuleEvaluationFactoryPtr, uint32 numThreads)
        : lossFactoryPtr_(std::move(lossFactoryPtr)),
          evaluationMeasureFactoryPtr_(std::move(evaluationMeasureFactoryPtr)),
          regularRuleEvaluationFactoryPtr_(std::move(regularRuleEvaluationFactoryPtr)),
          pruningRuleEvaluationFactoryPtr_(std::move(pruningRuleEvaluationFactoryPtr)), numThreads_(numThreads) {}

    std::unique_ptr<IStatisticsProvider> SparseLabelWiseStatisticsProviderFactory::create(
      const CContiguousView<const uint8>& labelMatrix) const {
        std::unique_ptr<ILabelWiseStatistics<ISparseLabelWiseRuleEvaluationFactory>> statisticsPtr = createStatistics(
          *lossFactoryPtr_, *evaluationMeasureFactoryPtr_, *regularRuleEvaluationFactoryPtr_, numThreads_, labelMatrix);
        return std::make_unique<LabelWiseStatisticsProvider<ISparseLabelWiseRuleEvaluationFactory>>(
          *regularRuleEvaluationFactoryPtr_, *pruningRuleEvaluationFactoryPtr_, std::move(statisticsPtr));
    }

    std::unique_ptr<IStatisticsProvider> SparseLabelWiseStatisticsProviderFactory::create(
      const BinaryCsrView& labelMatrix) const {
        std::unique_ptr<ILabelWiseStatistics<ISparseLabelWiseRuleEvaluationFactory>> statisticsPtr = createStatistics(
          *lossFactoryPtr_, *evaluationMeasureFactoryPtr_, *regularRuleEvaluationFactoryPtr_, numThreads_, labelMatrix);
        return std::make_unique<LabelWiseStatisticsProvider<ISparseLabelWiseRuleEvaluationFactory>>(
          *regularRuleEvaluationFactoryPtr_, *pruningRuleEvaluationFactoryPtr_, std::move(statisticsPtr));
    }
}

#ifdef _WIN32
    #pragma warning(pop)
#endif
