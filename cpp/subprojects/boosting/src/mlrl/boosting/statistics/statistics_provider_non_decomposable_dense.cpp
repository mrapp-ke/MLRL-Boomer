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
                          View<float64>::const_iterator hessiansEnd, uint32 weight) {
                util::addToViewWeighted(this->view.firstView.values_begin(row), gradientsBegin,
                                        this->view.firstView.numCols, weight);
                util::addToViewWeighted(this->view.secondView.values_begin(row), hessiansBegin,
                                        this->view.secondView.numCols, weight);
            }
    };

    /**
     * Provides access to gradients and Hessians that have been calculated according to a non-decomposable loss function
     * and are stored using dense data structures.
     *
     * @tparam Loss               The type of the non-decomposable loss function
     * @tparam OutputMatrix       The type of the matrix that provides access to the ground truth of the training
     *                            examples
     * @tparam EvaluationMeasure  The type of the evaluation measure that should be used to access the quality of
     *                            predictions
     */
    template<typename Loss, typename OutputMatrix, typename EvaluationMeasure>
    class DenseNonDecomposableStatistics final
        : public AbstractNonDecomposableStatistics<
            OutputMatrix, DenseNonDecomposableStatisticVector, DenseNonDecomposableStatisticMatrix,
            NumericCContiguousMatrix<float64>, Loss, EvaluationMeasure, INonDecomposableRuleEvaluationFactory,
            IDecomposableRuleEvaluationFactory> {
        public:

            /**
             * @param lossPtr               An unique pointer to an object of template type `Loss` that implements the
             *                              loss function to be used for calculating gradients and Hessians
             * @param evaluationMeasurePtr  An unique pointer to an object of template type `EvaluationMeasure` that
             *                              implements the evaluation measure that should be used to assess the quality
             *                              of predictions
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
            DenseNonDecomposableStatistics(std::unique_ptr<Loss> lossPtr,
                                           std::unique_ptr<EvaluationMeasure> evaluationMeasurePtr,
                                           const INonDecomposableRuleEvaluationFactory& ruleEvaluationFactory,
                                           const OutputMatrix& outputMatrix,
                                           std::unique_ptr<DenseNonDecomposableStatisticMatrix> statisticMatrixPtr,
                                           std::unique_ptr<NumericCContiguousMatrix<float64>> scoreMatrixPtr)
                : AbstractNonDecomposableStatistics<
                    OutputMatrix, DenseNonDecomposableStatisticVector, DenseNonDecomposableStatisticMatrix,
                    NumericCContiguousMatrix<float64>, Loss, EvaluationMeasure, INonDecomposableRuleEvaluationFactory,
                    IDecomposableRuleEvaluationFactory>(std::move(lossPtr), std::move(evaluationMeasurePtr),
                                                        ruleEvaluationFactory, outputMatrix,
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
              const IDecomposableRuleEvaluationFactory& ruleEvaluationFactory,
              MultiThreadingSettings multiThreadingSettings) override final {
                uint32 numRows = this->statisticMatrixPtr_->getNumRows();
                uint32 numCols = this->statisticMatrixPtr_->getNumCols();
                std::unique_ptr<DenseDecomposableStatisticMatrix> decomposableStatisticMatrixPtr =
                  std::make_unique<DenseDecomposableStatisticMatrix>(numRows, numCols);
                CContiguousView<Statistic<float64>>* decomposableStatisticMatrixRawPtr =
                  &decomposableStatisticMatrixPtr->getView();
                DenseNonDecomposableStatisticView* NonDecomposableStatisticViewRawPtr =
                  &this->statisticMatrixPtr_->getView();

#if MULTI_THREADING_SUPPORT_ENABLED
    #pragma omp parallel for firstprivate(numRows) firstprivate(numCols) \
      firstprivate(decomposableStatisticMatrixRawPtr) firstprivate(NonDecomposableStatisticViewRawPtr) \
      schedule(dynamic) num_threads(numThreads)
#endif
                for (int64 i = 0; i < numRows; i++) {
                    CContiguousView<Statistic<float64>>::value_iterator iterator =
                      decomposableStatisticMatrixRawPtr->values_begin(i);
                    DenseNonDecomposableStatisticView::gradient_const_iterator gradientIterator =
                      NonDecomposableStatisticViewRawPtr->gradients_cbegin(i);
                    DenseNonDecomposableStatisticView::hessian_diagonal_const_iterator hessianIterator =
                      NonDecomposableStatisticViewRawPtr->hessians_diagonal_cbegin(i);

                    for (uint32 j = 0; j < numCols; j++) {
                        Statistic<float64>& statistic = iterator[j];
                        statistic.gradient = gradientIterator[j];
                        statistic.hessian = hessianIterator[j];
                    }
                }

                return std::make_unique<DenseDecomposableStatistics<Loss, OutputMatrix, EvaluationMeasure>>(
                  std::move(this->lossPtr_), std::move(this->evaluationMeasurePtr_), ruleEvaluationFactory,
                  this->outputMatrix_, std::move(decomposableStatisticMatrixPtr), std::move(this->scoreMatrixPtr_));
            }
    };

    template<typename Loss, typename OutputMatrix, typename EvaluationMeasure>
    static inline std::unique_ptr<
      INonDecomposableStatistics<INonDecomposableRuleEvaluationFactory, IDecomposableRuleEvaluationFactory>>
      createStatistics(std::unique_ptr<Loss> lossPtr, std::unique_ptr<EvaluationMeasure> evaluationMeasurePtr,
                       const INonDecomposableRuleEvaluationFactory& ruleEvaluationFactory,
                       MultiThreadingSettings multiThreadingSettings, const OutputMatrix& outputMatrix) {
        uint32 numExamples = outputMatrix.numRows;
        uint32 numOutputs = outputMatrix.numCols;
        std::unique_ptr<DenseNonDecomposableStatisticMatrix> statisticMatrixPtr =
          std::make_unique<DenseNonDecomposableStatisticMatrix>(numExamples, numOutputs);
        std::unique_ptr<NumericCContiguousMatrix<float64>> scoreMatrixPtr =
          std::make_unique<NumericCContiguousMatrix<float64>>(numExamples, numOutputs, true);
        const Loss* lossRawPtr = lossPtr.get();
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

        return std::make_unique<DenseNonDecomposableStatistics<Loss, OutputMatrix, EvaluationMeasure>>(
          std::move(lossPtr), std::move(evaluationMeasurePtr), ruleEvaluationFactory, outputMatrix,
          std::move(statisticMatrixPtr), std::move(scoreMatrixPtr));
    }

    DenseNonDecomposableClassificationStatisticsProviderFactory::
      DenseNonDecomposableClassificationStatisticsProviderFactory(
        std::unique_ptr<INonDecomposableClassificationLossFactory> lossFactoryPtr,
        std::unique_ptr<IClassificationEvaluationMeasureFactory> evaluationMeasureFactoryPtr,
        std::unique_ptr<INonDecomposableRuleEvaluationFactory> defaultRuleEvaluationFactoryPtr,
        std::unique_ptr<INonDecomposableRuleEvaluationFactory> regularRuleEvaluationFactoryPtr,
        std::unique_ptr<INonDecomposableRuleEvaluationFactory> pruningRuleEvaluationFactoryPtr,
        MultiThreadingSettings multiThreadingSettings)
        : lossFactoryPtr_(std::move(lossFactoryPtr)),
          evaluationMeasureFactoryPtr_(std::move(evaluationMeasureFactoryPtr)),
          defaultRuleEvaluationFactoryPtr_(std::move(defaultRuleEvaluationFactoryPtr)),
          regularRuleEvaluationFactoryPtr_(std::move(regularRuleEvaluationFactoryPtr)),
          pruningRuleEvaluationFactoryPtr_(std::move(pruningRuleEvaluationFactoryPtr)),
          multiThreadingSettings_(multiThreadingSettings) {}

    std::unique_ptr<IStatisticsProvider> DenseNonDecomposableClassificationStatisticsProviderFactory::create(
      const CContiguousView<const uint8>& labelMatrix) const {
        std::unique_ptr<INonDecomposableClassificationLoss> lossPtr =
          lossFactoryPtr_->createNonDecomposableClassificationLoss();
        std::unique_ptr<IClassificationEvaluationMeasure> evaluationMeasurePtr =
          evaluationMeasureFactoryPtr_->createClassificationEvaluationMeasure();
        std::unique_ptr<
          INonDecomposableStatistics<INonDecomposableRuleEvaluationFactory, IDecomposableRuleEvaluationFactory>>
          statisticsPtr = createStatistics(std::move(lossPtr), std::move(evaluationMeasurePtr),
                                           *defaultRuleEvaluationFactoryPtr_, multiThreadingSettings_, labelMatrix);
        return std::make_unique<
          NonDecomposableStatisticsProvider<INonDecomposableRuleEvaluationFactory, IDecomposableRuleEvaluationFactory>>(
          *regularRuleEvaluationFactoryPtr_, *pruningRuleEvaluationFactoryPtr_, std::move(statisticsPtr));
    }

    std::unique_ptr<IStatisticsProvider> DenseNonDecomposableClassificationStatisticsProviderFactory::create(
      const BinaryCsrView& labelMatrix) const {
        std::unique_ptr<INonDecomposableClassificationLoss> lossPtr =
          lossFactoryPtr_->createNonDecomposableClassificationLoss();
        std::unique_ptr<IClassificationEvaluationMeasure> evaluationMeasurePtr =
          evaluationMeasureFactoryPtr_->createClassificationEvaluationMeasure();
        std::unique_ptr<
          INonDecomposableStatistics<INonDecomposableRuleEvaluationFactory, IDecomposableRuleEvaluationFactory>>
          statisticsPtr = createStatistics(std::move(lossPtr), std::move(evaluationMeasurePtr),
                                           *defaultRuleEvaluationFactoryPtr_, multiThreadingSettings_, labelMatrix);
        return std::make_unique<
          NonDecomposableStatisticsProvider<INonDecomposableRuleEvaluationFactory, IDecomposableRuleEvaluationFactory>>(
          *regularRuleEvaluationFactoryPtr_, *pruningRuleEvaluationFactoryPtr_, std::move(statisticsPtr));
    }

    DenseNonDecomposableRegressionStatisticsProviderFactory::DenseNonDecomposableRegressionStatisticsProviderFactory(
      std::unique_ptr<INonDecomposableRegressionLossFactory> lossFactoryPtr,
      std::unique_ptr<IRegressionEvaluationMeasureFactory> evaluationMeasureFactoryPtr,
      std::unique_ptr<INonDecomposableRuleEvaluationFactory> defaultRuleEvaluationFactoryPtr,
      std::unique_ptr<INonDecomposableRuleEvaluationFactory> regularRuleEvaluationFactoryPtr,
      std::unique_ptr<INonDecomposableRuleEvaluationFactory> pruningRuleEvaluationFactoryPtr,
      MultiThreadingSettings multiThreadingSettings)
        : lossFactoryPtr_(std::move(lossFactoryPtr)),
          evaluationMeasureFactoryPtr_(std::move(evaluationMeasureFactoryPtr)),
          defaultRuleEvaluationFactoryPtr_(std::move(defaultRuleEvaluationFactoryPtr)),
          regularRuleEvaluationFactoryPtr_(std::move(regularRuleEvaluationFactoryPtr)),
          pruningRuleEvaluationFactoryPtr_(std::move(pruningRuleEvaluationFactoryPtr)),
          multiThreadingSettings_(multiThreadingSettings) {}

    std::unique_ptr<IStatisticsProvider> DenseNonDecomposableRegressionStatisticsProviderFactory::create(
      const CContiguousView<const float32>& regressionMatrix) const {
        std::unique_ptr<INonDecomposableRegressionLoss> lossPtr =
          lossFactoryPtr_->createNonDecomposableRegressionLoss();
        std::unique_ptr<IRegressionEvaluationMeasure> evaluationMeasurePtr =
          evaluationMeasureFactoryPtr_->createRegressionEvaluationMeasure();
        std::unique_ptr<
          INonDecomposableStatistics<INonDecomposableRuleEvaluationFactory, IDecomposableRuleEvaluationFactory>>
          statisticsPtr =
            createStatistics(std::move(lossPtr), std::move(evaluationMeasurePtr), *defaultRuleEvaluationFactoryPtr_,
                             multiThreadingSettings_, regressionMatrix);
        return std::make_unique<
          NonDecomposableStatisticsProvider<INonDecomposableRuleEvaluationFactory, IDecomposableRuleEvaluationFactory>>(
          *regularRuleEvaluationFactoryPtr_, *pruningRuleEvaluationFactoryPtr_, std::move(statisticsPtr));
    }

    std::unique_ptr<IStatisticsProvider> DenseNonDecomposableRegressionStatisticsProviderFactory::create(
      const CsrView<const float32>& regressionMatrix) const {
        std::unique_ptr<INonDecomposableRegressionLoss> lossPtr =
          lossFactoryPtr_->createNonDecomposableRegressionLoss();
        std::unique_ptr<IRegressionEvaluationMeasure> evaluationMeasurePtr =
          evaluationMeasureFactoryPtr_->createRegressionEvaluationMeasure();
        std::unique_ptr<
          INonDecomposableStatistics<INonDecomposableRuleEvaluationFactory, IDecomposableRuleEvaluationFactory>>
          statisticsPtr =
            createStatistics(std::move(lossPtr), std::move(evaluationMeasurePtr), *defaultRuleEvaluationFactoryPtr_,
                             multiThreadingSettings_, regressionMatrix);
        return std::make_unique<
          NonDecomposableStatisticsProvider<INonDecomposableRuleEvaluationFactory, IDecomposableRuleEvaluationFactory>>(
          *regularRuleEvaluationFactoryPtr_, *pruningRuleEvaluationFactoryPtr_, std::move(statisticsPtr));
    }

    DenseConvertibleNonDecomposableClassificationStatisticsProviderFactory::
      DenseConvertibleNonDecomposableClassificationStatisticsProviderFactory(
        std::unique_ptr<INonDecomposableClassificationLossFactory> lossFactoryPtr,
        std::unique_ptr<IClassificationEvaluationMeasureFactory> evaluationMeasureFactoryPtr,
        std::unique_ptr<INonDecomposableRuleEvaluationFactory> defaultRuleEvaluationFactoryPtr,
        std::unique_ptr<IDecomposableRuleEvaluationFactory> regularRuleEvaluationFactoryPtr,
        std::unique_ptr<IDecomposableRuleEvaluationFactory> pruningRuleEvaluationFactoryPtr,
        MultiThreadingSettings multiThreadingSettings)
        : lossFactoryPtr_(std::move(lossFactoryPtr)),
          evaluationMeasureFactoryPtr_(std::move(evaluationMeasureFactoryPtr)),
          defaultRuleEvaluationFactoryPtr_(std::move(defaultRuleEvaluationFactoryPtr)),
          regularRuleEvaluationFactoryPtr_(std::move(regularRuleEvaluationFactoryPtr)),
          pruningRuleEvaluationFactoryPtr_(std::move(pruningRuleEvaluationFactoryPtr)),
          multiThreadingSettings_(multiThreadingSettings) {}

    std::unique_ptr<IStatisticsProvider> DenseConvertibleNonDecomposableClassificationStatisticsProviderFactory::create(
      const CContiguousView<const uint8>& labelMatrix) const {
        std::unique_ptr<INonDecomposableClassificationLoss> lossPtr =
          lossFactoryPtr_->createNonDecomposableClassificationLoss();
        std::unique_ptr<IClassificationEvaluationMeasure> evaluationMeasurePtr =
          evaluationMeasureFactoryPtr_->createClassificationEvaluationMeasure();
        std::unique_ptr<
          INonDecomposableStatistics<INonDecomposableRuleEvaluationFactory, IDecomposableRuleEvaluationFactory>>
          statisticsPtr = createStatistics(std::move(lossPtr), std::move(evaluationMeasurePtr),
                                           *defaultRuleEvaluationFactoryPtr_, multiThreadingSettings_, labelMatrix);
        return std::make_unique<ConvertibleNonDecomposableStatisticsProvider<INonDecomposableRuleEvaluationFactory,
                                                                             IDecomposableRuleEvaluationFactory>>(
          *regularRuleEvaluationFactoryPtr_, *pruningRuleEvaluationFactoryPtr_, std::move(statisticsPtr),
          multiThreadingSettings_);
    }

    std::unique_ptr<IStatisticsProvider> DenseConvertibleNonDecomposableClassificationStatisticsProviderFactory::create(
      const BinaryCsrView& labelMatrix) const {
        std::unique_ptr<INonDecomposableClassificationLoss> lossPtr =
          lossFactoryPtr_->createNonDecomposableClassificationLoss();
        std::unique_ptr<IClassificationEvaluationMeasure> evaluationMeasurePtr =
          evaluationMeasureFactoryPtr_->createClassificationEvaluationMeasure();
        std::unique_ptr<
          INonDecomposableStatistics<INonDecomposableRuleEvaluationFactory, IDecomposableRuleEvaluationFactory>>
          statisticsPtr = createStatistics(std::move(lossPtr), std::move(evaluationMeasurePtr),
                                           *defaultRuleEvaluationFactoryPtr_, multiThreadingSettings_, labelMatrix);
        return std::make_unique<ConvertibleNonDecomposableStatisticsProvider<INonDecomposableRuleEvaluationFactory,
                                                                             IDecomposableRuleEvaluationFactory>>(
          *regularRuleEvaluationFactoryPtr_, *pruningRuleEvaluationFactoryPtr_, std::move(statisticsPtr),
          multiThreadingSettings_);
    }

    DenseConvertibleNonDecomposableRegressionStatisticsProviderFactory::
      DenseConvertibleNonDecomposableRegressionStatisticsProviderFactory(
        std::unique_ptr<INonDecomposableRegressionLossFactory> lossFactoryPtr,
        std::unique_ptr<IRegressionEvaluationMeasureFactory> evaluationMeasureFactoryPtr,
        std::unique_ptr<INonDecomposableRuleEvaluationFactory> defaultRuleEvaluationFactoryPtr,
        std::unique_ptr<IDecomposableRuleEvaluationFactory> regularRuleEvaluationFactoryPtr,
        std::unique_ptr<IDecomposableRuleEvaluationFactory> pruningRuleEvaluationFactoryPtr,
        MultiThreadingSettings multiThreadingSettings)
        : lossFactoryPtr_(std::move(lossFactoryPtr)),
          evaluationMeasureFactoryPtr_(std::move(evaluationMeasureFactoryPtr)),
          defaultRuleEvaluationFactoryPtr_(std::move(defaultRuleEvaluationFactoryPtr)),
          regularRuleEvaluationFactoryPtr_(std::move(regularRuleEvaluationFactoryPtr)),
          pruningRuleEvaluationFactoryPtr_(std::move(pruningRuleEvaluationFactoryPtr)),
          multiThreadingSettings_(multiThreadingSettings) {}

    std::unique_ptr<IStatisticsProvider> DenseConvertibleNonDecomposableRegressionStatisticsProviderFactory::create(
      const CContiguousView<const float32>& regressionMatrix) const {
        std::unique_ptr<INonDecomposableRegressionLoss> lossPtr =
          lossFactoryPtr_->createNonDecomposableRegressionLoss();
        std::unique_ptr<IRegressionEvaluationMeasure> evaluationMeasurePtr =
          evaluationMeasureFactoryPtr_->createRegressionEvaluationMeasure();
        std::unique_ptr<
          INonDecomposableStatistics<INonDecomposableRuleEvaluationFactory, IDecomposableRuleEvaluationFactory>>
          statisticsPtr =
            createStatistics(std::move(lossPtr), std::move(evaluationMeasurePtr), *defaultRuleEvaluationFactoryPtr_,
                             multiThreadingSettings_, regressionMatrix);
        return std::make_unique<ConvertibleNonDecomposableStatisticsProvider<INonDecomposableRuleEvaluationFactory,
                                                                             IDecomposableRuleEvaluationFactory>>(
          *regularRuleEvaluationFactoryPtr_, *pruningRuleEvaluationFactoryPtr_, std::move(statisticsPtr),
          multiThreadingSettings_);
    }

    std::unique_ptr<IStatisticsProvider> DenseConvertibleNonDecomposableRegressionStatisticsProviderFactory::create(
      const CsrView<const float32>& regressionMatrix) const {
        std::unique_ptr<INonDecomposableRegressionLoss> lossPtr =
          lossFactoryPtr_->createNonDecomposableRegressionLoss();
        std::unique_ptr<IRegressionEvaluationMeasure> evaluationMeasurePtr =
          evaluationMeasureFactoryPtr_->createRegressionEvaluationMeasure();
        std::unique_ptr<
          INonDecomposableStatistics<INonDecomposableRuleEvaluationFactory, IDecomposableRuleEvaluationFactory>>
          statisticsPtr =
            createStatistics(std::move(lossPtr), std::move(evaluationMeasurePtr), *defaultRuleEvaluationFactoryPtr_,
                             multiThreadingSettings_, regressionMatrix);
        return std::make_unique<ConvertibleNonDecomposableStatisticsProvider<INonDecomposableRuleEvaluationFactory,
                                                                             IDecomposableRuleEvaluationFactory>>(
          *regularRuleEvaluationFactoryPtr_, *pruningRuleEvaluationFactoryPtr_, std::move(statisticsPtr),
          multiThreadingSettings_);
    }

}
