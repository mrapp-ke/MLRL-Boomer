#ifdef _WIN32
    #pragma warning(push)
    #pragma warning(disable : 4250)
#endif

#include "mlrl/boosting/statistics/statistics_provider_non_decomposable_dense.hpp"

#include "mlrl/boosting/data/matrix_c_contiguous_numeric.hpp"
#include "mlrl/boosting/data/vector_statistic_non_decomposable_dense.hpp"
#include "mlrl/boosting/data/view_statistic_non_decomposable_dense.hpp"
#include "mlrl/common/util/openmp.hpp"
#include "statistics_decomposable_dense.hpp"
#include "statistics_non_decomposable_common.hpp"
#include "statistics_provider_non_decomposable.hpp"

namespace boosting {

    /**
     * A matrix that stores gradients and Hessians that have been calculated using a non-decomposable loss function
     * using C-contiguous arrays.
     */
    class DenseNonDecomposableStatisticMatrix final
        : public ClearableViewDecorator<MatrixDecorator<DenseNonDecomposableStatisticView>> {
        public:

            /**
             * @param numRows The number of rows in the matrix
             * @param numCols The number of columns in the matrix
             */
            DenseNonDecomposableStatisticMatrix(uint32 numRows, uint32 numCols)
                : ClearableViewDecorator<MatrixDecorator<DenseNonDecomposableStatisticView>>(
                    DenseNonDecomposableStatisticView(numRows, numCols)) {}

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
     * Provides access to gradients and Hessians that have been calculated according to a non-decomposable loss function
     * and are stored using dense data structures.
     *
     * @tparam OutputMatrix The type of the matrix that provides access to the ground truth of the training examples
     */
    template<typename OutputMatrix>
    class DenseNonDecomposableStatistics final
        : public AbstractNonDecomposableStatistics<
            OutputMatrix, DenseNonDecomposableStatisticVector, DenseNonDecomposableStatisticMatrix,
            NumericCContiguousMatrix<float64>, INonDecomposableLoss, IEvaluationMeasure,
            INonDecomposableRuleEvaluationFactory, IDecomposableRuleEvaluationFactory> {
        public:

            /**
             * @param lossPtr               An unique pointer to an object of type `INonDecomposableLoss` that
             *                              implements the loss function to be used for calculating gradients and
             *                              Hessians
             * @param evaluationMeasurePtr  An unique pointer to an object of type `IEvaluationMeasure` that implements
             *                              the evaluation measure that should be used to assess the quality of
             *                              predictions
             * @param ruleEvaluationFactory A reference to an object of type `INonDecomposableRuleEvaluationFactory`, to
             *                              be used for calculating the predictions of rules, as well as their overall
             *                              quality
             * @param outputMatrix          A reference to an object of template type `OutputMatrix` that provides
             *                              access to the ground truth of the training examples
             * @param statisticMatrixPtr    An unique pointer to an object of type `DenseNonDecomposableStatisticMatrix`
             *                              that stores to the gradients and Hessians
             * @param scoreMatrixPtr        An unique pointer to an object of type `NumericCContiguousMatrix` that
             *                              stores the currently predicted scores
             */
            DenseNonDecomposableStatistics(std::unique_ptr<INonDecomposableLoss> lossPtr,
                                           std::unique_ptr<IEvaluationMeasure> evaluationMeasurePtr,
                                           const INonDecomposableRuleEvaluationFactory& ruleEvaluationFactory,
                                           const OutputMatrix& outputMatrix,
                                           std::unique_ptr<DenseNonDecomposableStatisticMatrix> statisticMatrixPtr,
                                           std::unique_ptr<NumericCContiguousMatrix<float64>> scoreMatrixPtr)
                : AbstractNonDecomposableStatistics<
                    OutputMatrix, DenseNonDecomposableStatisticVector, DenseNonDecomposableStatisticMatrix,
                    NumericCContiguousMatrix<float64>, INonDecomposableLoss, IEvaluationMeasure,
                    INonDecomposableRuleEvaluationFactory, IDecomposableRuleEvaluationFactory>(
                    std::move(lossPtr), std::move(evaluationMeasurePtr), ruleEvaluationFactory, outputMatrix,
                    std::move(statisticMatrixPtr), std::move(scoreMatrixPtr)) {}

            /**
             * @see `IBoostingStatistics::visitScoreMatrix`
             */
            void visitScoreMatrix(IBoostingStatistics::DenseScoreMatrixVisitor denseVisitor,
                                  IBoostingStatistics::SparseScoreMatrixVisitor sparseVisitor) const override {
                denseVisitor(this->scoreMatrixPtr_->getView());
            }

            /**
             * @see `INonDecomposableStatistics::toDecomposableStatistics`
             */
            std::unique_ptr<IDecomposableStatistics<IDecomposableRuleEvaluationFactory>> toDecomposableStatistics(
              const IDecomposableRuleEvaluationFactory& ruleEvaluationFactory, uint32 numThreads) override final {
                uint32 numRows = this->statisticMatrixPtr_->getNumRows();
                uint32 numCols = this->statisticMatrixPtr_->getNumCols();
                std::unique_ptr<DenseDecomposableStatisticMatrix> decomposableStatisticMatrixPtr =
                  std::make_unique<DenseDecomposableStatisticMatrix>(numRows, numCols);
                CContiguousView<Tuple<float64>>* decomposableStatisticMatrixRawPtr =
                  &decomposableStatisticMatrixPtr->getView();
                DenseNonDecomposableStatisticView* NonDecomposableStatisticViewRawPtr =
                  &this->statisticMatrixPtr_->getView();

#if MULTI_THREADING_SUPPORT_ENABLED
    #pragma omp parallel for firstprivate(numRows) firstprivate(numCols) \
      firstprivate(decomposableStatisticMatrixRawPtr) firstprivate(NonDecomposableStatisticViewRawPtr) \
      schedule(dynamic) num_threads(numThreads)
#endif
                for (int64 i = 0; i < numRows; i++) {
                    CContiguousView<Tuple<float64>>::value_iterator iterator =
                      decomposableStatisticMatrixRawPtr->values_begin(i);
                    DenseNonDecomposableStatisticView::gradient_const_iterator gradientIterator =
                      NonDecomposableStatisticViewRawPtr->gradients_cbegin(i);
                    DenseNonDecomposableStatisticView::hessian_diagonal_const_iterator hessianIterator =
                      NonDecomposableStatisticViewRawPtr->hessians_diagonal_cbegin(i);

                    for (uint32 j = 0; j < numCols; j++) {
                        Tuple<float64>& tuple = iterator[j];
                        tuple.first = gradientIterator[j];
                        tuple.second = hessianIterator[j];
                    }
                }

                return std::make_unique<DenseDecomposableStatistics<OutputMatrix>>(
                  std::move(this->lossPtr_), std::move(this->evaluationMeasurePtr_), ruleEvaluationFactory,
                  this->outputMatrix_, std::move(decomposableStatisticMatrixPtr), std::move(this->scoreMatrixPtr_));
            }
    };

    template<typename OutputMatrix>
    static inline std::unique_ptr<
      INonDecomposableStatistics<INonDecomposableRuleEvaluationFactory, IDecomposableRuleEvaluationFactory>>
      createStatistics(const INonDecomposableLossFactory& lossFactory,
                       const IEvaluationMeasureFactory& evaluationMeasureFactory,
                       const INonDecomposableRuleEvaluationFactory& ruleEvaluationFactory, uint32 numThreads,
                       const OutputMatrix& outputMatrix) {
        uint32 numExamples = outputMatrix.numRows;
        uint32 numOutputs = outputMatrix.numCols;
        std::unique_ptr<INonDecomposableLoss> lossPtr = lossFactory.createNonDecomposableLoss();
        std::unique_ptr<IEvaluationMeasure> evaluationMeasurePtr = evaluationMeasureFactory.createEvaluationMeasure();
        std::unique_ptr<DenseNonDecomposableStatisticMatrix> statisticMatrixPtr =
          std::make_unique<DenseNonDecomposableStatisticMatrix>(numExamples, numOutputs);
        std::unique_ptr<NumericCContiguousMatrix<float64>> scoreMatrixPtr =
          std::make_unique<NumericCContiguousMatrix<float64>>(numExamples, numOutputs, true);
        const INonDecomposableLoss* lossRawPtr = lossPtr.get();
        const OutputMatrix* outputMatrixPtr = &outputMatrix;
        const CContiguousView<float64>* scoreMatrixRawPtr = &scoreMatrixPtr->getView();
        DenseNonDecomposableStatisticView* statisticMatrixRawPtr = &statisticMatrixPtr->getView();

#if MULTI_THREADING_SUPPORT_ENABLED
    #pragma omp parallel for firstprivate(numExamples) firstprivate(lossRawPtr) firstprivate(outputMatrixPtr) \
      firstprivate(scoreMatrixRawPtr) firstprivate(statisticMatrixRawPtr) schedule(dynamic) num_threads(numThreads)
#endif
        for (int64 i = 0; i < numExamples; i++) {
            lossRawPtr->updateNonDecomposableStatistics(i, *outputMatrixPtr, *scoreMatrixRawPtr,
                                                        *statisticMatrixRawPtr);
        }

        return std::make_unique<DenseNonDecomposableStatistics<OutputMatrix>>(
          std::move(lossPtr), std::move(evaluationMeasurePtr), ruleEvaluationFactory, outputMatrix,
          std::move(statisticMatrixPtr), std::move(scoreMatrixPtr));
    }

    DenseNonDecomposableStatisticsProviderFactory::DenseNonDecomposableStatisticsProviderFactory(
      std::unique_ptr<INonDecomposableLossFactory> lossFactoryPtr,
      std::unique_ptr<IEvaluationMeasureFactory> evaluationMeasureFactoryPtr,
      std::unique_ptr<INonDecomposableRuleEvaluationFactory> defaultRuleEvaluationFactoryPtr,
      std::unique_ptr<INonDecomposableRuleEvaluationFactory> regularRuleEvaluationFactoryPtr,
      std::unique_ptr<INonDecomposableRuleEvaluationFactory> pruningRuleEvaluationFactoryPtr, uint32 numThreads)
        : lossFactoryPtr_(std::move(lossFactoryPtr)),
          evaluationMeasureFactoryPtr_(std::move(evaluationMeasureFactoryPtr)),
          defaultRuleEvaluationFactoryPtr_(std::move(defaultRuleEvaluationFactoryPtr)),
          regularRuleEvaluationFactoryPtr_(std::move(regularRuleEvaluationFactoryPtr)),
          pruningRuleEvaluationFactoryPtr_(std::move(pruningRuleEvaluationFactoryPtr)), numThreads_(numThreads) {}

    std::unique_ptr<IStatisticsProvider> DenseNonDecomposableStatisticsProviderFactory::create(
      const CContiguousView<const uint8>& labelMatrix) const {
        std::unique_ptr<
          INonDecomposableStatistics<INonDecomposableRuleEvaluationFactory, IDecomposableRuleEvaluationFactory>>
          statisticsPtr = createStatistics(*lossFactoryPtr_, *evaluationMeasureFactoryPtr_,
                                           *defaultRuleEvaluationFactoryPtr_, numThreads_, labelMatrix);
        return std::make_unique<
          NonDecomposableStatisticsProvider<INonDecomposableRuleEvaluationFactory, IDecomposableRuleEvaluationFactory>>(
          *regularRuleEvaluationFactoryPtr_, *pruningRuleEvaluationFactoryPtr_, std::move(statisticsPtr));
    }

    std::unique_ptr<IStatisticsProvider> DenseNonDecomposableStatisticsProviderFactory::create(
      const BinaryCsrView& labelMatrix) const {
        std::unique_ptr<
          INonDecomposableStatistics<INonDecomposableRuleEvaluationFactory, IDecomposableRuleEvaluationFactory>>
          statisticsPtr = createStatistics(*lossFactoryPtr_, *evaluationMeasureFactoryPtr_,
                                           *defaultRuleEvaluationFactoryPtr_, numThreads_, labelMatrix);
        return std::make_unique<
          NonDecomposableStatisticsProvider<INonDecomposableRuleEvaluationFactory, IDecomposableRuleEvaluationFactory>>(
          *regularRuleEvaluationFactoryPtr_, *pruningRuleEvaluationFactoryPtr_, std::move(statisticsPtr));
    }

    std::unique_ptr<IStatisticsProvider> DenseNonDecomposableStatisticsProviderFactory::create(
      const CContiguousView<const float32>& regressionMatrix) const {
        // TODO
        return nullptr;
    }

    std::unique_ptr<IStatisticsProvider> DenseNonDecomposableStatisticsProviderFactory::create(
      const CsrView<const float32>& regressionMatrix) const {
        // TODO
        return nullptr;
    }

    DenseConvertibleNonDecomposableStatisticsProviderFactory::DenseConvertibleNonDecomposableStatisticsProviderFactory(
      std::unique_ptr<INonDecomposableLossFactory> lossFactoryPtr,
      std::unique_ptr<IEvaluationMeasureFactory> evaluationMeasureFactoryPtr,
      std::unique_ptr<INonDecomposableRuleEvaluationFactory> defaultRuleEvaluationFactoryPtr,
      std::unique_ptr<IDecomposableRuleEvaluationFactory> regularRuleEvaluationFactoryPtr,
      std::unique_ptr<IDecomposableRuleEvaluationFactory> pruningRuleEvaluationFactoryPtr, uint32 numThreads)
        : lossFactoryPtr_(std::move(lossFactoryPtr)),
          evaluationMeasureFactoryPtr_(std::move(evaluationMeasureFactoryPtr)),
          defaultRuleEvaluationFactoryPtr_(std::move(defaultRuleEvaluationFactoryPtr)),
          regularRuleEvaluationFactoryPtr_(std::move(regularRuleEvaluationFactoryPtr)),
          pruningRuleEvaluationFactoryPtr_(std::move(pruningRuleEvaluationFactoryPtr)), numThreads_(numThreads) {}

    std::unique_ptr<IStatisticsProvider> DenseConvertibleNonDecomposableStatisticsProviderFactory::create(
      const CContiguousView<const uint8>& labelMatrix) const {
        std::unique_ptr<
          INonDecomposableStatistics<INonDecomposableRuleEvaluationFactory, IDecomposableRuleEvaluationFactory>>
          statisticsPtr = createStatistics(*lossFactoryPtr_, *evaluationMeasureFactoryPtr_,
                                           *defaultRuleEvaluationFactoryPtr_, numThreads_, labelMatrix);
        return std::make_unique<ConvertibleNonDecomposableStatisticsProvider<INonDecomposableRuleEvaluationFactory,
                                                                             IDecomposableRuleEvaluationFactory>>(
          *regularRuleEvaluationFactoryPtr_, *pruningRuleEvaluationFactoryPtr_, std::move(statisticsPtr), numThreads_);
    }

    std::unique_ptr<IStatisticsProvider> DenseConvertibleNonDecomposableStatisticsProviderFactory::create(
      const BinaryCsrView& labelMatrix) const {
        std::unique_ptr<
          INonDecomposableStatistics<INonDecomposableRuleEvaluationFactory, IDecomposableRuleEvaluationFactory>>
          statisticsPtr = createStatistics(*lossFactoryPtr_, *evaluationMeasureFactoryPtr_,
                                           *defaultRuleEvaluationFactoryPtr_, numThreads_, labelMatrix);
        return std::make_unique<ConvertibleNonDecomposableStatisticsProvider<INonDecomposableRuleEvaluationFactory,
                                                                             IDecomposableRuleEvaluationFactory>>(
          *regularRuleEvaluationFactoryPtr_, *pruningRuleEvaluationFactoryPtr_, std::move(statisticsPtr), numThreads_);
    }

    std::unique_ptr<IStatisticsProvider> DenseConvertibleNonDecomposableStatisticsProviderFactory::create(
      const CContiguousView<const float32>& regressionMatrix) const {
        // TODO
        return nullptr;
    }

    std::unique_ptr<IStatisticsProvider> DenseConvertibleNonDecomposableStatisticsProviderFactory::create(
      const CsrView<const float32>& regressionMatrix) const {
        // TODO
        return nullptr;
    }

}

#ifdef _WIN32
    #pragma warning(pop)
#endif
