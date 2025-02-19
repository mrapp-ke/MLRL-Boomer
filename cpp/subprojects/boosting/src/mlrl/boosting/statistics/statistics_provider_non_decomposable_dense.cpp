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
     *
     * @tparam StatisticType The type of the gradients and Hessians
     */
    template<typename StatisticType>
    class DenseNonDecomposableStatisticMatrix final
        : public ClearableViewDecorator<MatrixDecorator<DenseNonDecomposableStatisticView<StatisticType>>> {
        public:

            /**
             * @param numRows The number of rows in the matrix
             * @param numCols The number of columns in the matrix
             */
            DenseNonDecomposableStatisticMatrix(uint32 numRows, uint32 numCols)
                : ClearableViewDecorator<MatrixDecorator<DenseNonDecomposableStatisticView<StatisticType>>>(
                    DenseNonDecomposableStatisticView<StatisticType>(numRows, numCols)) {}

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
            OutputMatrix, DenseNonDecomposableStatisticMatrix<typename Loss::statistic_type>,
            NumericCContiguousMatrix<typename Loss::statistic_type>, Loss, EvaluationMeasure,
            INonDecomposableRuleEvaluationFactory, IDecomposableRuleEvaluationFactory> {
        private:

            typedef typename Loss::statistic_type statistic_type;

            typedef NonDecomposableStatisticsState<OutputMatrix, DenseNonDecomposableStatisticMatrix<statistic_type>,
                                                   NumericCContiguousMatrix<statistic_type>, Loss>
              StatisticsState;

            template<typename WeightVector, typename IndexVector>
            using StatisticsSubset =
              StatisticsSubset<StatisticsState, DenseNonDecomposableStatisticVector<statistic_type>,
                               INonDecomposableRuleEvaluationFactory, WeightVector, IndexVector>;

            template<typename WeightVector>
            using WeightedStatistics =
              WeightedStatistics<StatisticsState, DenseNonDecomposableStatisticVector<statistic_type>,
                                 INonDecomposableRuleEvaluationFactory, WeightVector>;

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
            DenseNonDecomposableStatistics(
              std::unique_ptr<Loss> lossPtr, std::unique_ptr<EvaluationMeasure> evaluationMeasurePtr,
              const INonDecomposableRuleEvaluationFactory& ruleEvaluationFactory, const OutputMatrix& outputMatrix,
              std::unique_ptr<DenseNonDecomposableStatisticMatrix<statistic_type>> statisticMatrixPtr,
              std::unique_ptr<NumericCContiguousMatrix<statistic_type>> scoreMatrixPtr)
                : AbstractNonDecomposableStatistics<OutputMatrix, DenseNonDecomposableStatisticMatrix<statistic_type>,
                                                    NumericCContiguousMatrix<statistic_type>, Loss, EvaluationMeasure,
                                                    INonDecomposableRuleEvaluationFactory,
                                                    IDecomposableRuleEvaluationFactory>(
                    std::move(lossPtr), std::move(evaluationMeasurePtr), ruleEvaluationFactory, outputMatrix,
                    std::move(statisticMatrixPtr), std::move(scoreMatrixPtr)) {}

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(const CompleteIndexVector& outputIndices,
                                                            const EqualWeightVector& weights) const override {
                return std::make_unique<StatisticsSubset<EqualWeightVector, CompleteIndexVector>>(
                  *this->statePtr_, *this->ruleEvaluationFactory_, weights, outputIndices);
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(const PartialIndexVector& outputIndices,
                                                            const EqualWeightVector& weights) const override {
                return std::make_unique<StatisticsSubset<EqualWeightVector, PartialIndexVector>>(
                  *this->statePtr_, *this->ruleEvaluationFactory_, weights, outputIndices);
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(const CompleteIndexVector& outputIndices,
                                                            const BitWeightVector& weights) const override {
                return std::make_unique<StatisticsSubset<BitWeightVector, CompleteIndexVector>>(
                  *this->statePtr_, *this->ruleEvaluationFactory_, weights, outputIndices);
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(const PartialIndexVector& outputIndices,
                                                            const BitWeightVector& weights) const override {
                return std::make_unique<StatisticsSubset<BitWeightVector, PartialIndexVector>>(
                  *this->statePtr_, *this->ruleEvaluationFactory_, weights, outputIndices);
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(const CompleteIndexVector& outputIndices,
                                                            const DenseWeightVector<uint32>& weights) const override {
                return std::make_unique<StatisticsSubset<DenseWeightVector<uint32>, CompleteIndexVector>>(
                  *this->statePtr_, *this->ruleEvaluationFactory_, weights, outputIndices);
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(const PartialIndexVector& outputIndices,
                                                            const DenseWeightVector<uint32>& weights) const override {
                return std::make_unique<StatisticsSubset<DenseWeightVector<uint32>, PartialIndexVector>>(
                  *this->statePtr_, *this->ruleEvaluationFactory_, weights, outputIndices);
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(
              const CompleteIndexVector& outputIndices,
              const DenseWeightVector<float32>& weights) const override final {
                return std::make_unique<StatisticsSubset<DenseWeightVector<float32>, CompleteIndexVector>>(
                  *this->statePtr_, *this->ruleEvaluationFactory_, weights, outputIndices);
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(const PartialIndexVector& outputIndices,
                                                            const DenseWeightVector<float32>& weights) const override {
                return std::make_unique<StatisticsSubset<DenseWeightVector<float32>, PartialIndexVector>>(
                  *this->statePtr_, *this->ruleEvaluationFactory_, weights, outputIndices);
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(
              const CompleteIndexVector& outputIndices,
              const OutOfSampleWeightVector<EqualWeightVector>& weights) const override {
                return std::make_unique<
                  StatisticsSubset<OutOfSampleWeightVector<EqualWeightVector>, CompleteIndexVector>>(
                  *this->statePtr_, *this->ruleEvaluationFactory_, weights, outputIndices);
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(
              const PartialIndexVector& outputIndices,
              const OutOfSampleWeightVector<EqualWeightVector>& weights) const override {
                return std::make_unique<
                  StatisticsSubset<OutOfSampleWeightVector<EqualWeightVector>, PartialIndexVector>>(
                  *this->statePtr_, *this->ruleEvaluationFactory_, weights, outputIndices);
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(
              const CompleteIndexVector& outputIndices,
              const OutOfSampleWeightVector<BitWeightVector>& weights) const override {
                return std::make_unique<
                  StatisticsSubset<OutOfSampleWeightVector<BitWeightVector>, CompleteIndexVector>>(
                  *this->statePtr_, *this->ruleEvaluationFactory_, weights, outputIndices);
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(
              const PartialIndexVector& outputIndices,
              const OutOfSampleWeightVector<BitWeightVector>& weights) const override {
                return std::make_unique<StatisticsSubset<OutOfSampleWeightVector<BitWeightVector>, PartialIndexVector>>(
                  *this->statePtr_, *this->ruleEvaluationFactory_, weights, outputIndices);
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(
              const CompleteIndexVector& outputIndices,
              const OutOfSampleWeightVector<DenseWeightVector<uint32>>& weights) const override {
                return std::make_unique<
                  StatisticsSubset<OutOfSampleWeightVector<DenseWeightVector<uint32>>, CompleteIndexVector>>(
                  *this->statePtr_, *this->ruleEvaluationFactory_, weights, outputIndices);
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(
              const PartialIndexVector& outputIndices,
              const OutOfSampleWeightVector<DenseWeightVector<uint32>>& weights) const override {
                return std::make_unique<
                  StatisticsSubset<OutOfSampleWeightVector<DenseWeightVector<uint32>>, PartialIndexVector>>(
                  *this->statePtr_, *this->ruleEvaluationFactory_, weights, outputIndices);
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(
              const CompleteIndexVector& outputIndices,
              const OutOfSampleWeightVector<DenseWeightVector<float32>>& weights) const override {
                return std::make_unique<
                  StatisticsSubset<OutOfSampleWeightVector<DenseWeightVector<float32>>, CompleteIndexVector>>(
                  *this->statePtr_, *this->ruleEvaluationFactory_, weights, outputIndices);
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(
              const PartialIndexVector& outputIndices,
              const OutOfSampleWeightVector<DenseWeightVector<float32>>& weights) const override {
                return std::make_unique<
                  StatisticsSubset<OutOfSampleWeightVector<DenseWeightVector<float32>>, PartialIndexVector>>(
                  *this->statePtr_, *this->ruleEvaluationFactory_, weights, outputIndices);
            }

            /**
             * @see `IStatistics::createWeightedStatistics`
             */
            std::unique_ptr<IWeightedStatistics> createWeightedStatistics(
              const EqualWeightVector& weights) const override {
                return std::make_unique<WeightedStatistics<EqualWeightVector>>(*this->statePtr_,
                                                                               *this->ruleEvaluationFactory_, weights);
            }

            /**
             * @see `IStatistics::createWeightedStatistics`
             */
            std::unique_ptr<IWeightedStatistics> createWeightedStatistics(
              const BitWeightVector& weights) const override {
                return std::make_unique<WeightedStatistics<BitWeightVector>>(*this->statePtr_,
                                                                             *this->ruleEvaluationFactory_, weights);
            }

            /**
             * @see `IStatistics::createWeightedStatistics`
             */
            std::unique_ptr<IWeightedStatistics> createWeightedStatistics(
              const DenseWeightVector<uint32>& weights) const override {
                return std::make_unique<WeightedStatistics<DenseWeightVector<uint32>>>(
                  *this->statePtr_, *this->ruleEvaluationFactory_, weights);
            }

            /**
             * @see `IStatistics::createWeightedStatistics`
             */
            std::unique_ptr<IWeightedStatistics> createWeightedStatistics(
              const DenseWeightVector<float32>& weights) const override {
                return std::make_unique<WeightedStatistics<DenseWeightVector<float32>>>(
                  *this->statePtr_, *this->ruleEvaluationFactory_, weights);
            }

            /**
             * @see `IBoostingStatistics::visitScoreMatrix`
             */
            void visitScoreMatrix(
              IBoostingStatistics::DenseScoreMatrixVisitor<float32> dense32BitVisitor,
              IBoostingStatistics::DenseScoreMatrixVisitor<float64> dense64BitVisitor,
              IBoostingStatistics::SparseScoreMatrixVisitor<float32> sparse32BitVisitor,
              IBoostingStatistics::SparseScoreMatrixVisitor<float64> sparse64BitVisitor) const override {
                visitScoreMatrixInternally(this->statePtr_->scoreMatrixPtr->getView(), dense32BitVisitor,
                                           dense64BitVisitor);
            }

            /**
             * @see `INonDecomposableStatistics::toDecomposableStatistics`
             */
            std::unique_ptr<IDecomposableStatistics<IDecomposableRuleEvaluationFactory>> toDecomposableStatistics(
              const IDecomposableRuleEvaluationFactory& ruleEvaluationFactory,
              MultiThreadingSettings multiThreadingSettings) override final {
                uint32 numRows = this->statePtr_->statisticMatrixPtr->getNumRows();
                uint32 numCols = this->statePtr_->statisticMatrixPtr->getNumCols();
                std::unique_ptr<DenseDecomposableStatisticMatrix<statistic_type>> decomposableStatisticMatrixPtr =
                  std::make_unique<DenseDecomposableStatisticMatrix<statistic_type>>(numRows, numCols);
                CContiguousView<Statistic<statistic_type>>* decomposableStatisticMatrixRawPtr =
                  &decomposableStatisticMatrixPtr->getView();
                DenseNonDecomposableStatisticView<statistic_type>* NonDecomposableStatisticViewRawPtr =
                  &this->statePtr_->statisticMatrixPtr->getView();

#if MULTI_THREADING_SUPPORT_ENABLED
    #pragma omp parallel for firstprivate(numRows) firstprivate(numCols) \
      firstprivate(decomposableStatisticMatrixRawPtr) firstprivate(NonDecomposableStatisticViewRawPtr) \
      schedule(dynamic) num_threads(numThreads)
#endif
                for (int64 i = 0; i < numRows; i++) {
                    typename CContiguousView<Statistic<statistic_type>>::value_iterator iterator =
                      decomposableStatisticMatrixRawPtr->values_begin(i);
                    typename DenseNonDecomposableStatisticView<statistic_type>::gradient_const_iterator
                      gradientIterator = NonDecomposableStatisticViewRawPtr->gradients_cbegin(i);
                    typename DenseNonDecomposableStatisticView<statistic_type>::hessian_diagonal_const_iterator
                      hessianIterator = NonDecomposableStatisticViewRawPtr->hessians_diagonal_cbegin(i);

                    for (uint32 j = 0; j < numCols; j++) {
                        Statistic<statistic_type>& statistic = iterator[j];
                        statistic.gradient = gradientIterator[j];
                        statistic.hessian = hessianIterator[j];
                    }
                }

                return std::make_unique<DenseDecomposableStatistics<Loss, OutputMatrix, EvaluationMeasure>>(
                  std::move(this->statePtr_->lossFunctionPtr), std::move(this->evaluationMeasurePtr_),
                  ruleEvaluationFactory, this->statePtr_->outputMatrix, std::move(decomposableStatisticMatrixPtr),
                  std::move(this->statePtr_->scoreMatrixPtr));
            }
    };

    template<typename Loss, typename OutputMatrix, typename EvaluationMeasure>
    static inline std::unique_ptr<
      INonDecomposableStatistics<INonDecomposableRuleEvaluationFactory, IDecomposableRuleEvaluationFactory>>
      createStatistics(std::unique_ptr<Loss> lossPtr, std::unique_ptr<EvaluationMeasure> evaluationMeasurePtr,
                       const INonDecomposableRuleEvaluationFactory& ruleEvaluationFactory,
                       MultiThreadingSettings multiThreadingSettings, const OutputMatrix& outputMatrix) {
        typedef typename Loss::statistic_type statistic_type;
        uint32 numExamples = outputMatrix.numRows;
        uint32 numOutputs = outputMatrix.numCols;
        std::unique_ptr<DenseNonDecomposableStatisticMatrix<statistic_type>> statisticMatrixPtr =
          std::make_unique<DenseNonDecomposableStatisticMatrix<statistic_type>>(numExamples, numOutputs);
        std::unique_ptr<NumericCContiguousMatrix<statistic_type>> scoreMatrixPtr =
          std::make_unique<NumericCContiguousMatrix<statistic_type>>(numExamples, numOutputs, true);
        const Loss* lossRawPtr = lossPtr.get();
        const OutputMatrix* outputMatrixPtr = &outputMatrix;
        const CContiguousView<statistic_type>* scoreMatrixRawPtr = &scoreMatrixPtr->getView();
        DenseNonDecomposableStatisticView<statistic_type>* statisticMatrixRawPtr = &statisticMatrixPtr->getView();

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

    template<typename StatisticType>
    DenseNonDecomposableClassificationStatisticsProviderFactory<StatisticType>::
      DenseNonDecomposableClassificationStatisticsProviderFactory(
        std::unique_ptr<INonDecomposableClassificationLossFactory<StatisticType>> lossFactoryPtr,
        std::unique_ptr<IClassificationEvaluationMeasureFactory<StatisticType>> evaluationMeasureFactoryPtr,
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

    template<typename StatisticType>
    std::unique_ptr<IStatisticsProvider>
      DenseNonDecomposableClassificationStatisticsProviderFactory<StatisticType>::create(
        const CContiguousView<const uint8>& labelMatrix) const {
        std::unique_ptr<INonDecomposableClassificationLoss<StatisticType>> lossPtr =
          lossFactoryPtr_->createNonDecomposableClassificationLoss();
        std::unique_ptr<IClassificationEvaluationMeasure<StatisticType>> evaluationMeasurePtr =
          evaluationMeasureFactoryPtr_->createClassificationEvaluationMeasure();
        std::unique_ptr<
          INonDecomposableStatistics<INonDecomposableRuleEvaluationFactory, IDecomposableRuleEvaluationFactory>>
          statisticsPtr = createStatistics(std::move(lossPtr), std::move(evaluationMeasurePtr),
                                           *defaultRuleEvaluationFactoryPtr_, multiThreadingSettings_, labelMatrix);
        return std::make_unique<
          NonDecomposableStatisticsProvider<INonDecomposableRuleEvaluationFactory, IDecomposableRuleEvaluationFactory>>(
          *regularRuleEvaluationFactoryPtr_, *pruningRuleEvaluationFactoryPtr_, std::move(statisticsPtr));
    }

    template<typename StatisticType>
    std::unique_ptr<IStatisticsProvider>
      DenseNonDecomposableClassificationStatisticsProviderFactory<StatisticType>::create(
        const BinaryCsrView& labelMatrix) const {
        std::unique_ptr<INonDecomposableClassificationLoss<StatisticType>> lossPtr =
          lossFactoryPtr_->createNonDecomposableClassificationLoss();
        std::unique_ptr<IClassificationEvaluationMeasure<StatisticType>> evaluationMeasurePtr =
          evaluationMeasureFactoryPtr_->createClassificationEvaluationMeasure();
        std::unique_ptr<
          INonDecomposableStatistics<INonDecomposableRuleEvaluationFactory, IDecomposableRuleEvaluationFactory>>
          statisticsPtr = createStatistics(std::move(lossPtr), std::move(evaluationMeasurePtr),
                                           *defaultRuleEvaluationFactoryPtr_, multiThreadingSettings_, labelMatrix);
        return std::make_unique<
          NonDecomposableStatisticsProvider<INonDecomposableRuleEvaluationFactory, IDecomposableRuleEvaluationFactory>>(
          *regularRuleEvaluationFactoryPtr_, *pruningRuleEvaluationFactoryPtr_, std::move(statisticsPtr));
    }

    template class DenseNonDecomposableClassificationStatisticsProviderFactory<float64>;

    template<typename StatisticType>
    DenseNonDecomposableRegressionStatisticsProviderFactory<StatisticType>::
      DenseNonDecomposableRegressionStatisticsProviderFactory(
        std::unique_ptr<INonDecomposableRegressionLossFactory<StatisticType>> lossFactoryPtr,
        std::unique_ptr<IRegressionEvaluationMeasureFactory<StatisticType>> evaluationMeasureFactoryPtr,
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

    template<typename StatisticType>
    std::unique_ptr<IStatisticsProvider> DenseNonDecomposableRegressionStatisticsProviderFactory<StatisticType>::create(
      const CContiguousView<const float32>& regressionMatrix) const {
        std::unique_ptr<INonDecomposableRegressionLoss<StatisticType>> lossPtr =
          lossFactoryPtr_->createNonDecomposableRegressionLoss();
        std::unique_ptr<IRegressionEvaluationMeasure<StatisticType>> evaluationMeasurePtr =
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

    template<typename StatisticType>
    std::unique_ptr<IStatisticsProvider> DenseNonDecomposableRegressionStatisticsProviderFactory<StatisticType>::create(
      const CsrView<const float32>& regressionMatrix) const {
        std::unique_ptr<INonDecomposableRegressionLoss<StatisticType>> lossPtr =
          lossFactoryPtr_->createNonDecomposableRegressionLoss();
        std::unique_ptr<IRegressionEvaluationMeasure<StatisticType>> evaluationMeasurePtr =
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

    template class DenseNonDecomposableRegressionStatisticsProviderFactory<float64>;

    template<typename StatisticType>
    DenseConvertibleNonDecomposableClassificationStatisticsProviderFactory<StatisticType>::
      DenseConvertibleNonDecomposableClassificationStatisticsProviderFactory(
        std::unique_ptr<INonDecomposableClassificationLossFactory<StatisticType>> lossFactoryPtr,
        std::unique_ptr<IClassificationEvaluationMeasureFactory<StatisticType>> evaluationMeasureFactoryPtr,
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

    template<typename StatisticType>
    std::unique_ptr<IStatisticsProvider>
      DenseConvertibleNonDecomposableClassificationStatisticsProviderFactory<StatisticType>::create(
        const CContiguousView<const uint8>& labelMatrix) const {
        std::unique_ptr<INonDecomposableClassificationLoss<StatisticType>> lossPtr =
          lossFactoryPtr_->createNonDecomposableClassificationLoss();
        std::unique_ptr<IClassificationEvaluationMeasure<StatisticType>> evaluationMeasurePtr =
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

    template<typename StatisticType>
    std::unique_ptr<IStatisticsProvider>
      DenseConvertibleNonDecomposableClassificationStatisticsProviderFactory<StatisticType>::create(
        const BinaryCsrView& labelMatrix) const {
        std::unique_ptr<INonDecomposableClassificationLoss<StatisticType>> lossPtr =
          lossFactoryPtr_->createNonDecomposableClassificationLoss();
        std::unique_ptr<IClassificationEvaluationMeasure<StatisticType>> evaluationMeasurePtr =
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

    template class DenseConvertibleNonDecomposableClassificationStatisticsProviderFactory<float64>;

    template<typename StatisticType>
    DenseConvertibleNonDecomposableRegressionStatisticsProviderFactory<StatisticType>::
      DenseConvertibleNonDecomposableRegressionStatisticsProviderFactory(
        std::unique_ptr<INonDecomposableRegressionLossFactory<StatisticType>> lossFactoryPtr,
        std::unique_ptr<IRegressionEvaluationMeasureFactory<StatisticType>> evaluationMeasureFactoryPtr,
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

    template<typename StatisticType>
    std::unique_ptr<IStatisticsProvider>
      DenseConvertibleNonDecomposableRegressionStatisticsProviderFactory<StatisticType>::create(
        const CContiguousView<const float32>& regressionMatrix) const {
        std::unique_ptr<INonDecomposableRegressionLoss<StatisticType>> lossPtr =
          lossFactoryPtr_->createNonDecomposableRegressionLoss();
        std::unique_ptr<IRegressionEvaluationMeasure<StatisticType>> evaluationMeasurePtr =
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

    template<typename StatisticType>
    std::unique_ptr<IStatisticsProvider>
      DenseConvertibleNonDecomposableRegressionStatisticsProviderFactory<StatisticType>::create(
        const CsrView<const float32>& regressionMatrix) const {
        std::unique_ptr<INonDecomposableRegressionLoss<StatisticType>> lossPtr =
          lossFactoryPtr_->createNonDecomposableRegressionLoss();
        std::unique_ptr<IRegressionEvaluationMeasure<StatisticType>> evaluationMeasurePtr =
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

    template class DenseConvertibleNonDecomposableRegressionStatisticsProviderFactory<float64>;

}
