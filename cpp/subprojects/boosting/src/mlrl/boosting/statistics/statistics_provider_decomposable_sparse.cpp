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
     *
     * @tparam StatisticType the type of the gradients and Hessians
     */
    template<typename StatisticType>
    class SparseDecomposableStatisticMatrix final : public MatrixDecorator<SparseSetView<Statistic<StatisticType>>> {
        public:

            /**
             * @param numRows   The number of rows in the matrix
             * @param numCols   The number of columns in the matrix
             */
            SparseDecomposableStatisticMatrix(uint32 numRows, uint32 numCols)
                : MatrixDecorator<SparseSetView<Statistic<StatisticType>>>(
                    SparseSetView<Statistic<StatisticType>>(numRows, numCols)) {}
    };

    static inline void visitScoreMatrixInternally(
      const SparseSetView<float32>& view, IBoostingStatistics::SparseScoreMatrixVisitor<float32> sparse32BitVisitor,
      IBoostingStatistics::SparseScoreMatrixVisitor<float64> sparse64BitVisitor) {
        sparse32BitVisitor(view);
    }

    static inline void visitScoreMatrixInternally(
      const SparseSetView<float64>& view, IBoostingStatistics::SparseScoreMatrixVisitor<float32> sparse32BitVisitor,
      IBoostingStatistics::SparseScoreMatrixVisitor<float64> sparse64BitVisitor) {
        sparse64BitVisitor(view);
    }

    /**
     * Provides access to gradients and Hessians that have been calculated according to a decomposable loss function
     * and are stored using sparse data structures.
     *
     * @tparam Loss                 The type of the loss function
     * @tparam OutputMatrix         The type of the matrix that provides access to the ground truth of the training
     *                              examples
     * @tparam EvaluationMeasure    The type of the evaluation that should be used to access the quality of predictions
     */
    template<typename Loss, typename OutputMatrix, typename EvaluationMeasure>
    class SparseDecomposableStatistics final
        : public AbstractDecomposableStatistics<OutputMatrix,
                                                SparseDecomposableStatisticMatrix<typename Loss::statistic_type>,
                                                NumericSparseSetMatrix<typename Loss::statistic_type>, Loss,
                                                EvaluationMeasure, ISparseDecomposableRuleEvaluationFactory> {
        private:

            typedef typename Loss::statistic_type statistic_type;

            typedef DecomposableBoostingStatisticsState<OutputMatrix, SparseDecomposableStatisticMatrix<statistic_type>,
                                                        NumericSparseSetMatrix<statistic_type>, Loss>
              StatisticsState;

            template<typename WeightVector, typename IndexVector, typename WeightType>
            using StatisticsSubset =
              StatisticsSubset<StatisticsState, SparseDecomposableStatisticVector<statistic_type, WeightType>,
                               ISparseDecomposableRuleEvaluationFactory, WeightVector, IndexVector>;

            template<typename WeightVector, typename WeightType>
            using WeightedStatistics =
              WeightedStatistics<StatisticsState, SparseDecomposableStatisticVector<statistic_type, WeightType>,
                                 ISparseDecomposableRuleEvaluationFactory, WeightVector>;

        public:

            /**
             * @param lossPtr               An unique pointer to an object of template type `Loss` that implements the
             *                              loss function that should be used for calculating gradients and Hessians
             * @param evaluationMeasurePtr  An unique pointer to an object of template type `EvaluationMeasure` that
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
            SparseDecomposableStatistics(
              std::unique_ptr<Loss> lossPtr, std::unique_ptr<EvaluationMeasure> evaluationMeasurePtr,
              const ISparseDecomposableRuleEvaluationFactory& ruleEvaluationFactory, const OutputMatrix& outputMatrix,
              std::unique_ptr<SparseDecomposableStatisticMatrix<statistic_type>> statisticViewPtr,
              std::unique_ptr<NumericSparseSetMatrix<statistic_type>> scoreMatrixPtr)
                : AbstractDecomposableStatistics<OutputMatrix, SparseDecomposableStatisticMatrix<statistic_type>,
                                                 NumericSparseSetMatrix<statistic_type>, Loss, EvaluationMeasure,
                                                 ISparseDecomposableRuleEvaluationFactory>(
                    std::move(lossPtr), std::move(evaluationMeasurePtr), ruleEvaluationFactory, outputMatrix,
                    std::move(statisticViewPtr), std::move(scoreMatrixPtr)) {}

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(const CompleteIndexVector& outputIndices,
                                                            const EqualWeightVector& weights) const override {
                return std::make_unique<StatisticsSubset<EqualWeightVector, CompleteIndexVector, uint32>>(
                  *this->statePtr_, *this->ruleEvaluationFactory_, weights, outputIndices);
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(const PartialIndexVector& outputIndices,
                                                            const EqualWeightVector& weights) const override {
                return std::make_unique<StatisticsSubset<EqualWeightVector, PartialIndexVector, uint32>>(
                  *this->statePtr_, *this->ruleEvaluationFactory_, weights, outputIndices);
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(const CompleteIndexVector& outputIndices,
                                                            const BitWeightVector& weights) const override {
                return std::make_unique<StatisticsSubset<BitWeightVector, CompleteIndexVector, uint32>>(
                  *this->statePtr_, *this->ruleEvaluationFactory_, weights, outputIndices);
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(const PartialIndexVector& outputIndices,
                                                            const BitWeightVector& weights) const override {
                return std::make_unique<StatisticsSubset<BitWeightVector, PartialIndexVector, uint32>>(
                  *this->statePtr_, *this->ruleEvaluationFactory_, weights, outputIndices);
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(const CompleteIndexVector& outputIndices,
                                                            const DenseWeightVector<uint16>& weights) const override {
                return std::make_unique<StatisticsSubset<DenseWeightVector<uint16>, CompleteIndexVector, uint32>>(
                  *this->statePtr_, *this->ruleEvaluationFactory_, weights, outputIndices);
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(const PartialIndexVector& outputIndices,
                                                            const DenseWeightVector<uint16>& weights) const override {
                return std::make_unique<StatisticsSubset<DenseWeightVector<uint16>, PartialIndexVector, uint32>>(
                  *this->statePtr_, *this->ruleEvaluationFactory_, weights, outputIndices);
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(
              const CompleteIndexVector& outputIndices,
              const DenseWeightVector<float32>& weights) const override final {
                return std::make_unique<StatisticsSubset<DenseWeightVector<float32>, CompleteIndexVector, float32>>(
                  *this->statePtr_, *this->ruleEvaluationFactory_, weights, outputIndices);
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(const PartialIndexVector& outputIndices,
                                                            const DenseWeightVector<float32>& weights) const override {
                return std::make_unique<StatisticsSubset<DenseWeightVector<float32>, PartialIndexVector, float32>>(
                  *this->statePtr_, *this->ruleEvaluationFactory_, weights, outputIndices);
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(
              const CompleteIndexVector& outputIndices,
              const OutOfSampleWeightVector<EqualWeightVector>& weights) const override {
                return std::make_unique<
                  StatisticsSubset<OutOfSampleWeightVector<EqualWeightVector>, CompleteIndexVector, uint32>>(
                  *this->statePtr_, *this->ruleEvaluationFactory_, weights, outputIndices);
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(
              const PartialIndexVector& outputIndices,
              const OutOfSampleWeightVector<EqualWeightVector>& weights) const override {
                return std::make_unique<
                  StatisticsSubset<OutOfSampleWeightVector<EqualWeightVector>, PartialIndexVector, uint32>>(
                  *this->statePtr_, *this->ruleEvaluationFactory_, weights, outputIndices);
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(
              const CompleteIndexVector& outputIndices,
              const OutOfSampleWeightVector<BitWeightVector>& weights) const override {
                return std::make_unique<
                  StatisticsSubset<OutOfSampleWeightVector<BitWeightVector>, CompleteIndexVector, uint32>>(
                  *this->statePtr_, *this->ruleEvaluationFactory_, weights, outputIndices);
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(
              const PartialIndexVector& outputIndices,
              const OutOfSampleWeightVector<BitWeightVector>& weights) const override {
                return std::make_unique<
                  StatisticsSubset<OutOfSampleWeightVector<BitWeightVector>, PartialIndexVector, uint32>>(
                  *this->statePtr_, *this->ruleEvaluationFactory_, weights, outputIndices);
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(
              const CompleteIndexVector& outputIndices,
              const OutOfSampleWeightVector<DenseWeightVector<uint16>>& weights) const override {
                return std::make_unique<
                  StatisticsSubset<OutOfSampleWeightVector<DenseWeightVector<uint16>>, CompleteIndexVector, uint32>>(
                  *this->statePtr_, *this->ruleEvaluationFactory_, weights, outputIndices);
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(
              const PartialIndexVector& outputIndices,
              const OutOfSampleWeightVector<DenseWeightVector<uint16>>& weights) const override {
                return std::make_unique<
                  StatisticsSubset<OutOfSampleWeightVector<DenseWeightVector<uint16>>, PartialIndexVector, uint32>>(
                  *this->statePtr_, *this->ruleEvaluationFactory_, weights, outputIndices);
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(
              const CompleteIndexVector& outputIndices,
              const OutOfSampleWeightVector<DenseWeightVector<float32>>& weights) const override {
                return std::make_unique<
                  StatisticsSubset<OutOfSampleWeightVector<DenseWeightVector<float32>>, CompleteIndexVector, uint32>>(
                  *this->statePtr_, *this->ruleEvaluationFactory_, weights, outputIndices);
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(
              const PartialIndexVector& outputIndices,
              const OutOfSampleWeightVector<DenseWeightVector<float32>>& weights) const override {
                return std::make_unique<
                  StatisticsSubset<OutOfSampleWeightVector<DenseWeightVector<float32>>, PartialIndexVector, uint32>>(
                  *this->statePtr_, *this->ruleEvaluationFactory_, weights, outputIndices);
            }

            /**
             * @see `IStatistics::createWeightedStatistics`
             */
            std::unique_ptr<IWeightedStatistics> createWeightedStatistics(
              const EqualWeightVector& weights) const override {
                return std::make_unique<WeightedStatistics<EqualWeightVector, uint32>>(
                  *this->statePtr_, *this->ruleEvaluationFactory_, weights);
            }

            /**
             * @see `IStatistics::createWeightedStatistics`
             */
            std::unique_ptr<IWeightedStatistics> createWeightedStatistics(
              const BitWeightVector& weights) const override {
                return std::make_unique<WeightedStatistics<BitWeightVector, uint32>>(
                  *this->statePtr_, *this->ruleEvaluationFactory_, weights);
            }

            /**
             * @see `IStatistics::createWeightedStatistics`
             */
            std::unique_ptr<IWeightedStatistics> createWeightedStatistics(
              const DenseWeightVector<uint16>& weights) const override {
                return std::make_unique<WeightedStatistics<DenseWeightVector<uint16>, uint32>>(
                  *this->statePtr_, *this->ruleEvaluationFactory_, weights);
            }

            /**
             * @see `IStatistics::createWeightedStatistics`
             */
            std::unique_ptr<IWeightedStatistics> createWeightedStatistics(
              const DenseWeightVector<float32>& weights) const override {
                return std::make_unique<WeightedStatistics<DenseWeightVector<float32>, float32>>(
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
                visitScoreMatrixInternally(this->statePtr_->scoreMatrixPtr->getView(), sparse32BitVisitor,
                                           sparse64BitVisitor);
            }
    };

    template<typename Loss, typename OutputMatrix, typename EvaluationMeasure>
    static inline std::unique_ptr<IDecomposableStatistics<ISparseDecomposableRuleEvaluationFactory>> createStatistics(
      std::unique_ptr<Loss> lossPtr, std::unique_ptr<EvaluationMeasure> evaluationMeasurePtr,
      const ISparseDecomposableRuleEvaluationFactory& ruleEvaluationFactory,
      MultiThreadingSettings multiThreadingSettings, const OutputMatrix& outputMatrix) {
        typedef typename Loss::statistic_type statistic_type;
        uint32 numExamples = outputMatrix.numRows;
        uint32 numOutputs = outputMatrix.numCols;
        std::unique_ptr<SparseDecomposableStatisticMatrix<statistic_type>> statisticMatrixPtr =
          std::make_unique<SparseDecomposableStatisticMatrix<statistic_type>>(numExamples, numOutputs);
        std::unique_ptr<NumericSparseSetMatrix<statistic_type>> scoreMatrixPtr =
          std::make_unique<NumericSparseSetMatrix<statistic_type>>(numExamples, numOutputs);
        const Loss* lossRawPtr = lossPtr.get();
        const OutputMatrix* outputMatrixPtr = &outputMatrix;
        const SparseSetView<statistic_type>* scoreMatrixRawPtr = &scoreMatrixPtr->getView();
        SparseSetView<Statistic<statistic_type>>* statisticMatrixRawPtr = &statisticMatrixPtr->getView();

#if MULTI_THREADING_SUPPORT_ENABLED
    #pragma omp parallel for firstprivate(numExamples) firstprivate(lossRawPtr) firstprivate(outputMatrixPtr) \
      firstprivate(scoreMatrixRawPtr) firstprivate(statisticMatrixRawPtr) schedule(dynamic) \
      num_threads(multiThreadingSettings.numThreads)
#endif
        for (int64 i = 0; i < numExamples; i++) {
            lossRawPtr->updateDecomposableStatistics(i, *outputMatrixPtr, *scoreMatrixRawPtr, IndexIterator(),
                                                     IndexIterator(outputMatrixPtr->numCols), *statisticMatrixRawPtr);
        }

        return std::make_unique<SparseDecomposableStatistics<Loss, OutputMatrix, EvaluationMeasure>>(
          std::move(lossPtr), std::move(evaluationMeasurePtr), ruleEvaluationFactory, outputMatrix,
          std::move(statisticMatrixPtr), std::move(scoreMatrixPtr));
    }

    template<typename StatisticType>
    SparseDecomposableClassificationStatisticsProviderFactory<StatisticType>::
      SparseDecomposableClassificationStatisticsProviderFactory(
        std::unique_ptr<ISparseDecomposableClassificationLossFactory<StatisticType>> lossFactoryPtr,
        std::unique_ptr<ISparseEvaluationMeasureFactory<StatisticType>> evaluationMeasureFactoryPtr,
        std::unique_ptr<ISparseDecomposableRuleEvaluationFactory> regularRuleEvaluationFactoryPtr,
        std::unique_ptr<ISparseDecomposableRuleEvaluationFactory> pruningRuleEvaluationFactoryPtr,
        MultiThreadingSettings multiThreadingSettings)
        : lossFactoryPtr_(std::move(lossFactoryPtr)),
          evaluationMeasureFactoryPtr_(std::move(evaluationMeasureFactoryPtr)),
          regularRuleEvaluationFactoryPtr_(std::move(regularRuleEvaluationFactoryPtr)),
          pruningRuleEvaluationFactoryPtr_(std::move(pruningRuleEvaluationFactoryPtr)),
          multiThreadingSettings_(multiThreadingSettings) {}

    template<typename StatisticType>
    std::unique_ptr<IStatisticsProvider>
      SparseDecomposableClassificationStatisticsProviderFactory<StatisticType>::create(
        const CContiguousView<const uint8>& labelMatrix) const {
        std::unique_ptr<ISparseDecomposableClassificationLoss<StatisticType>> lossPtr =
          lossFactoryPtr_->createSparseDecomposableClassificationLoss();
        std::unique_ptr<ISparseEvaluationMeasure<StatisticType>> evaluationMeasurePtr =
          evaluationMeasureFactoryPtr_->createSparseEvaluationMeasure();
        std::unique_ptr<IDecomposableStatistics<ISparseDecomposableRuleEvaluationFactory>> statisticsPtr =
          createStatistics(std::move(lossPtr), std::move(evaluationMeasurePtr), *regularRuleEvaluationFactoryPtr_,
                           multiThreadingSettings_, labelMatrix);
        return std::make_unique<DecomposableStatisticsProvider<ISparseDecomposableRuleEvaluationFactory>>(
          *regularRuleEvaluationFactoryPtr_, *pruningRuleEvaluationFactoryPtr_, std::move(statisticsPtr));
    }

    template<typename StatisticType>
    std::unique_ptr<IStatisticsProvider>
      SparseDecomposableClassificationStatisticsProviderFactory<StatisticType>::create(
        const BinaryCsrView& labelMatrix) const {
        std::unique_ptr<ISparseDecomposableClassificationLoss<StatisticType>> lossPtr =
          lossFactoryPtr_->createSparseDecomposableClassificationLoss();
        std::unique_ptr<ISparseEvaluationMeasure<StatisticType>> evaluationMeasurePtr =
          evaluationMeasureFactoryPtr_->createSparseEvaluationMeasure();
        std::unique_ptr<IDecomposableStatistics<ISparseDecomposableRuleEvaluationFactory>> statisticsPtr =
          createStatistics(std::move(lossPtr), std::move(evaluationMeasurePtr), *regularRuleEvaluationFactoryPtr_,
                           multiThreadingSettings_, labelMatrix);
        return std::make_unique<DecomposableStatisticsProvider<ISparseDecomposableRuleEvaluationFactory>>(
          *regularRuleEvaluationFactoryPtr_, *pruningRuleEvaluationFactoryPtr_, std::move(statisticsPtr));
    }

    template class SparseDecomposableClassificationStatisticsProviderFactory<float32>;
    template class SparseDecomposableClassificationStatisticsProviderFactory<float64>;
}
