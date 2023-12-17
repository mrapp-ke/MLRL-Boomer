#ifdef _WIN32
    #pragma warning(push)
    #pragma warning(disable : 4250)
#endif

#include "mlrl/boosting/statistics/statistics_provider_example_wise_dense.hpp"

#include "mlrl/boosting/data/matrix_c_contiguous_numeric.hpp"
#include "mlrl/boosting/data/vector_statistic_example_wise_dense.hpp"
#include "mlrl/boosting/data/view_statistic_example_wise_dense.hpp"
#include "mlrl/common/util/openmp.hpp"
#include "statistics_example_wise_common.hpp"
#include "statistics_label_wise_dense.hpp"
#include "statistics_provider_example_wise.hpp"

namespace boosting {

    /**
     * A matrix that stores gradients and Hessians that have been calculated using a non-decomposable loss function
     * using C-contiguous arrays.
     */
    class DenseExampleWiseStatisticMatrix final
        : public ClearableViewDecorator<MatrixDecorator<DenseExampleWiseStatisticView>> {
        public:

            /**
             * @param numRows The number of rows in the matrix
             * @param numCols The number of columns in the matrix
             */
            DenseExampleWiseStatisticMatrix(uint32 numRows, uint32 numCols)
                : ClearableViewDecorator<MatrixDecorator<DenseExampleWiseStatisticView>>(
                  DenseExampleWiseStatisticView(numRows, numCols)) {}

            /**
             * Adds all gradients and Hessians in a vector to a specific row of this matrix. The gradients and Hessians
             * to be added are multiplied by a specific weight.
             *
             * @param row               The row
             * @param gradientsBegin    An iterator to the beginning of the gradients in the vector
             * @param gradientsEnd      An iterator to the end of the gradients in the vector
             * @param hessiansBegin     An iterator to the beginning of the Hessians in the vector
             * @param hessiansEnd       An iterator to the end of the Hessians in the vector
             * @param weight            The weight, the gradients and Hessians should be multiplied by
             */
            void addToRow(uint32 row, View<float64>::const_iterator gradientsBegin,
                          View<float64>::const_iterator gradientsEnd, View<float64>::const_iterator hessiansBegin,
                          View<float64>::const_iterator hessiansEnd, float64 weight) {
                addToView(this->view.firstView.values_begin(row), gradientsBegin, this->view.firstView.numCols, weight);
                addToView(this->view.secondView.values_begin(row), hessiansBegin, this->view.secondView.numCols,
                          weight);
            }
    };

    /**
     * Provides access to gradients and Hessians that have been calculated according to a differentiable loss function
     * that is applied example-wise and are stored using dense data structures.
     *
     * @tparam LabelMatrix The type of the matrix that provides access to the labels of the training examples
     */
    template<typename LabelMatrix>
    class DenseExampleWiseStatistics final
        : public AbstractExampleWiseStatistics<LabelMatrix, DenseExampleWiseStatisticVector,
                                               DenseExampleWiseStatisticMatrix, DenseExampleWiseStatisticMatrix,
                                               NumericCContiguousMatrix<float64>, IExampleWiseLoss, IEvaluationMeasure,
                                               IExampleWiseRuleEvaluationFactory, ILabelWiseRuleEvaluationFactory> {
        public:

            /**
             * @param lossPtr               An unique pointer to an object of type `IExampleWiseLoss` that implements
             *                              the loss function to be used for calculating gradients and Hessians
             * @param evaluationMeasurePtr  An unique pointer to an object of type `IEvaluationMeasure` that implements
             *                              the evaluation measure that should be used to assess the quality of
             *                              predictions
             * @param ruleEvaluationFactory A reference to an object of type `IExampleWiseRuleEvaluationFactory`, to be
             *                              used for calculating the predictions, as well as corresponding quality
             *                              scores, of rules
             * @param labelMatrix           A reference to an object of template type `LabelMatrix` that provides access
             *                              to the labels of the training examples
             * @param statisticMatrixPtr    An unique pointer to an object of type `DenseExampleWiseStatisticMatrix`
             *                              that stores to the gradients and Hessians
             * @param scoreMatrixPtr        An unique pointer to an object of type `NumericCContiguousMatrix` that
             *                              stores the currently predicted scores
             */
            DenseExampleWiseStatistics(std::unique_ptr<IExampleWiseLoss> lossPtr,
                                       std::unique_ptr<IEvaluationMeasure> evaluationMeasurePtr,
                                       const IExampleWiseRuleEvaluationFactory& ruleEvaluationFactory,
                                       const LabelMatrix& labelMatrix,
                                       std::unique_ptr<DenseExampleWiseStatisticMatrix> statisticMatrixPtr,
                                       std::unique_ptr<NumericCContiguousMatrix<float64>> scoreMatrixPtr)
                : AbstractExampleWiseStatistics<LabelMatrix, DenseExampleWiseStatisticVector,
                                                DenseExampleWiseStatisticMatrix, DenseExampleWiseStatisticMatrix,
                                                NumericCContiguousMatrix<float64>, IExampleWiseLoss, IEvaluationMeasure,
                                                IExampleWiseRuleEvaluationFactory, ILabelWiseRuleEvaluationFactory>(
                  std::move(lossPtr), std::move(evaluationMeasurePtr), ruleEvaluationFactory, labelMatrix,
                  std::move(statisticMatrixPtr), std::move(scoreMatrixPtr)) {}

            /**
             * @see `IBoostingStatistics::visitScoreMatrix`
             */
            void visitScoreMatrix(IBoostingStatistics::DenseScoreMatrixVisitor denseVisitor,
                                  IBoostingStatistics::SparseScoreMatrixVisitor sparseVisitor) const override {
                denseVisitor(this->scoreMatrixPtr_->getView());
            }

            /**
             * @see `IExampleWiseStatistics::toLabelWiseStatistics`
             */
            std::unique_ptr<ILabelWiseStatistics<ILabelWiseRuleEvaluationFactory>> toLabelWiseStatistics(
              const ILabelWiseRuleEvaluationFactory& ruleEvaluationFactory, uint32 numThreads) override final {
                uint32 numRows = this->statisticMatrixPtr_->getNumRows();
                uint32 numCols = this->statisticMatrixPtr_->getNumCols();
                std::unique_ptr<DenseLabelWiseStatisticMatrix> labelWiseStatisticMatrixPtr =
                  std::make_unique<DenseLabelWiseStatisticMatrix>(numRows, numCols);
                CContiguousView<Tuple<float64>>* labelWiseStatisticMatrixRawPtr =
                  &labelWiseStatisticMatrixPtr->getView();
                DenseExampleWiseStatisticView* exampleWiseStatisticViewRawPtr = &this->statisticMatrixPtr_->getView();

#if MULTI_THREADING_SUPPORT_ENABLED
    #pragma omp parallel for firstprivate(numRows) firstprivate(numCols) firstprivate(labelWiseStatisticMatrixRawPtr) \
      firstprivate(exampleWiseStatisticViewRawPtr) schedule(dynamic) num_threads(numThreads)
#endif
                for (int64 i = 0; i < numRows; i++) {
                    CContiguousView<Tuple<float64>>::value_iterator iterator =
                      labelWiseStatisticMatrixRawPtr->values_begin(i);
                    DenseExampleWiseStatisticView::gradient_const_iterator gradientIterator =
                      exampleWiseStatisticViewRawPtr->gradients_cbegin(i);
                    DenseExampleWiseStatisticView::hessian_diagonal_const_iterator hessianIterator =
                      exampleWiseStatisticViewRawPtr->hessians_diagonal_cbegin(i);

                    for (uint32 j = 0; j < numCols; j++) {
                        Tuple<float64>& tuple = iterator[j];
                        tuple.first = gradientIterator[j];
                        tuple.second = hessianIterator[j];
                    }
                }

                return std::make_unique<DenseLabelWiseStatistics<LabelMatrix>>(
                  std::move(this->lossPtr_), std::move(this->evaluationMeasurePtr_), ruleEvaluationFactory,
                  this->labelMatrix_, std::move(labelWiseStatisticMatrixPtr), std::move(this->scoreMatrixPtr_));
            }
    };

    template<typename LabelMatrix>
    static inline std::unique_ptr<
      IExampleWiseStatistics<IExampleWiseRuleEvaluationFactory, ILabelWiseRuleEvaluationFactory>>
      createStatistics(const IExampleWiseLossFactory& lossFactory,
                       const IEvaluationMeasureFactory& evaluationMeasureFactory,
                       const IExampleWiseRuleEvaluationFactory& ruleEvaluationFactory, uint32 numThreads,
                       const LabelMatrix& labelMatrix) {
        uint32 numExamples = labelMatrix.numRows;
        uint32 numLabels = labelMatrix.numCols;
        std::unique_ptr<IExampleWiseLoss> lossPtr = lossFactory.createExampleWiseLoss();
        std::unique_ptr<IEvaluationMeasure> evaluationMeasurePtr = evaluationMeasureFactory.createEvaluationMeasure();
        std::unique_ptr<DenseExampleWiseStatisticMatrix> statisticMatrixPtr =
          std::make_unique<DenseExampleWiseStatisticMatrix>(numExamples, numLabels);
        std::unique_ptr<NumericCContiguousMatrix<float64>> scoreMatrixPtr =
          std::make_unique<NumericCContiguousMatrix<float64>>(numExamples, numLabels, true);
        const IExampleWiseLoss* lossRawPtr = lossPtr.get();
        const LabelMatrix* labelMatrixPtr = &labelMatrix;
        const CContiguousView<float64>* scoreMatrixRawPtr = &scoreMatrixPtr->getView();
        DenseExampleWiseStatisticView* statisticMatrixRawPtr = &statisticMatrixPtr->getView();

#if MULTI_THREADING_SUPPORT_ENABLED
    #pragma omp parallel for firstprivate(numExamples) firstprivate(lossRawPtr) firstprivate(labelMatrixPtr) \
      firstprivate(scoreMatrixRawPtr) firstprivate(statisticMatrixRawPtr) schedule(dynamic) num_threads(numThreads)
#endif
        for (int64 i = 0; i < numExamples; i++) {
            lossRawPtr->updateExampleWiseStatistics(i, *labelMatrixPtr, *scoreMatrixRawPtr, *statisticMatrixRawPtr);
        }

        return std::make_unique<DenseExampleWiseStatistics<LabelMatrix>>(
          std::move(lossPtr), std::move(evaluationMeasurePtr), ruleEvaluationFactory, labelMatrix,
          std::move(statisticMatrixPtr), std::move(scoreMatrixPtr));
    }

    DenseExampleWiseStatisticsProviderFactory::DenseExampleWiseStatisticsProviderFactory(
      std::unique_ptr<IExampleWiseLossFactory> lossFactoryPtr,
      std::unique_ptr<IEvaluationMeasureFactory> evaluationMeasureFactoryPtr,
      std::unique_ptr<IExampleWiseRuleEvaluationFactory> defaultRuleEvaluationFactoryPtr,
      std::unique_ptr<IExampleWiseRuleEvaluationFactory> regularRuleEvaluationFactoryPtr,
      std::unique_ptr<IExampleWiseRuleEvaluationFactory> pruningRuleEvaluationFactoryPtr, uint32 numThreads)
        : lossFactoryPtr_(std::move(lossFactoryPtr)),
          evaluationMeasureFactoryPtr_(std::move(evaluationMeasureFactoryPtr)),
          defaultRuleEvaluationFactoryPtr_(std::move(defaultRuleEvaluationFactoryPtr)),
          regularRuleEvaluationFactoryPtr_(std::move(regularRuleEvaluationFactoryPtr)),
          pruningRuleEvaluationFactoryPtr_(std::move(pruningRuleEvaluationFactoryPtr)), numThreads_(numThreads) {}

    std::unique_ptr<IStatisticsProvider> DenseExampleWiseStatisticsProviderFactory::create(
      const CContiguousView<const uint8>& labelMatrix) const {
        std::unique_ptr<IExampleWiseStatistics<IExampleWiseRuleEvaluationFactory, ILabelWiseRuleEvaluationFactory>>
          statisticsPtr = createStatistics(*lossFactoryPtr_, *evaluationMeasureFactoryPtr_,
                                           *defaultRuleEvaluationFactoryPtr_, numThreads_, labelMatrix);
        return std::make_unique<
          ExampleWiseStatisticsProvider<IExampleWiseRuleEvaluationFactory, ILabelWiseRuleEvaluationFactory>>(
          *regularRuleEvaluationFactoryPtr_, *pruningRuleEvaluationFactoryPtr_, std::move(statisticsPtr));
    }

    std::unique_ptr<IStatisticsProvider> DenseExampleWiseStatisticsProviderFactory::create(
      const BinaryCsrView& labelMatrix) const {
        std::unique_ptr<IExampleWiseStatistics<IExampleWiseRuleEvaluationFactory, ILabelWiseRuleEvaluationFactory>>
          statisticsPtr = createStatistics(*lossFactoryPtr_, *evaluationMeasureFactoryPtr_,
                                           *defaultRuleEvaluationFactoryPtr_, numThreads_, labelMatrix);
        return std::make_unique<
          ExampleWiseStatisticsProvider<IExampleWiseRuleEvaluationFactory, ILabelWiseRuleEvaluationFactory>>(
          *regularRuleEvaluationFactoryPtr_, *pruningRuleEvaluationFactoryPtr_, std::move(statisticsPtr));
    }

    DenseConvertibleExampleWiseStatisticsProviderFactory::DenseConvertibleExampleWiseStatisticsProviderFactory(
      std::unique_ptr<IExampleWiseLossFactory> lossFactoryPtr,
      std::unique_ptr<IEvaluationMeasureFactory> evaluationMeasureFactoryPtr,
      std::unique_ptr<IExampleWiseRuleEvaluationFactory> defaultRuleEvaluationFactoryPtr,
      std::unique_ptr<ILabelWiseRuleEvaluationFactory> regularRuleEvaluationFactoryPtr,
      std::unique_ptr<ILabelWiseRuleEvaluationFactory> pruningRuleEvaluationFactoryPtr, uint32 numThreads)
        : lossFactoryPtr_(std::move(lossFactoryPtr)),
          evaluationMeasureFactoryPtr_(std::move(evaluationMeasureFactoryPtr)),
          defaultRuleEvaluationFactoryPtr_(std::move(defaultRuleEvaluationFactoryPtr)),
          regularRuleEvaluationFactoryPtr_(std::move(regularRuleEvaluationFactoryPtr)),
          pruningRuleEvaluationFactoryPtr_(std::move(pruningRuleEvaluationFactoryPtr)), numThreads_(numThreads) {}

    std::unique_ptr<IStatisticsProvider> DenseConvertibleExampleWiseStatisticsProviderFactory::create(
      const CContiguousView<const uint8>& labelMatrix) const {
        std::unique_ptr<IExampleWiseStatistics<IExampleWiseRuleEvaluationFactory, ILabelWiseRuleEvaluationFactory>>
          statisticsPtr = createStatistics(*lossFactoryPtr_, *evaluationMeasureFactoryPtr_,
                                           *defaultRuleEvaluationFactoryPtr_, numThreads_, labelMatrix);
        return std::make_unique<
          ConvertibleExampleWiseStatisticsProvider<IExampleWiseRuleEvaluationFactory, ILabelWiseRuleEvaluationFactory>>(
          *regularRuleEvaluationFactoryPtr_, *pruningRuleEvaluationFactoryPtr_, std::move(statisticsPtr), numThreads_);
    }

    std::unique_ptr<IStatisticsProvider> DenseConvertibleExampleWiseStatisticsProviderFactory::create(
      const BinaryCsrView& labelMatrix) const {
        std::unique_ptr<IExampleWiseStatistics<IExampleWiseRuleEvaluationFactory, ILabelWiseRuleEvaluationFactory>>
          statisticsPtr = createStatistics(*lossFactoryPtr_, *evaluationMeasureFactoryPtr_,
                                           *defaultRuleEvaluationFactoryPtr_, numThreads_, labelMatrix);
        return std::make_unique<
          ConvertibleExampleWiseStatisticsProvider<IExampleWiseRuleEvaluationFactory, ILabelWiseRuleEvaluationFactory>>(
          *regularRuleEvaluationFactoryPtr_, *pruningRuleEvaluationFactoryPtr_, std::move(statisticsPtr), numThreads_);
    }

}

#ifdef _WIN32
    #pragma warning(pop)
#endif
