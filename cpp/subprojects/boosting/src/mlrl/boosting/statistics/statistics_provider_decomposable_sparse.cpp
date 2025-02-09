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
    class SparseDecomposableStatisticMatrix final : public MatrixDecorator<SparseSetView<Statistic<float64>>> {
        public:

            /**
             * @param numRows   The number of rows in the matrix
             * @param numCols   The number of columns in the matrix
             */
            SparseDecomposableStatisticMatrix(uint32 numRows, uint32 numCols)
                : MatrixDecorator<SparseSetView<Statistic<float64>>>(
                    SparseSetView<Statistic<float64>>(numRows, numCols)) {}
    };

    /**
     * Provides access to gradients and Hessians that have been calculated according to a decomposable loss function
     * and are stored using sparse data structures.
     *
     * @tparam OutputMatrix The type of the matrix that provides access to the ground truth of the training examples
     */
    template<typename OutputMatrix>
    class SparseDecomposableStatistics final
        : public AbstractDecomposableStatistics<OutputMatrix, SparseDecomposableStatisticMatrix,
                                                NumericSparseSetMatrix<float64>, ISparseDecomposableClassificationLoss,
                                                ISparseEvaluationMeasure, ISparseDecomposableRuleEvaluationFactory> {
        private:

            using StatisticsState =
              DecomposableStatisticsState<OutputMatrix, SparseDecomposableStatisticMatrix,
                                          NumericSparseSetMatrix<float64>, ISparseDecomposableClassificationLoss>;

            template<typename WeightVector, typename IndexVector, typename WeightType>
            using StatisticsSubset =
              StatisticsSubset<StatisticsState, SparseDecomposableStatisticVector<WeightType>,
                               ISparseDecomposableRuleEvaluationFactory, WeightVector, IndexVector>;

            template<typename WeightVector, typename WeightType>
            using WeightedStatistics =
              WeightedStatistics<StatisticsState, SparseDecomposableStatisticVector<WeightType>,
                                 ISparseDecomposableRuleEvaluationFactory, WeightVector>;

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
            SparseDecomposableStatistics(std::unique_ptr<ISparseDecomposableClassificationLoss> lossPtr,
                                         std::unique_ptr<ISparseEvaluationMeasure> evaluationMeasurePtr,
                                         const ISparseDecomposableRuleEvaluationFactory& ruleEvaluationFactory,
                                         const OutputMatrix& outputMatrix,
                                         std::unique_ptr<SparseDecomposableStatisticMatrix> statisticViewPtr,
                                         std::unique_ptr<NumericSparseSetMatrix<float64>> scoreMatrixPtr)
                : AbstractDecomposableStatistics<OutputMatrix, SparseDecomposableStatisticMatrix,
                                                 NumericSparseSetMatrix<float64>, ISparseDecomposableClassificationLoss,
                                                 ISparseEvaluationMeasure, ISparseDecomposableRuleEvaluationFactory>(
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
                                                            const DenseWeightVector<uint32>& weights) const override {
                return std::make_unique<StatisticsSubset<DenseWeightVector<uint32>, CompleteIndexVector, uint32>>(
                  *this->statePtr_, *this->ruleEvaluationFactory_, weights, outputIndices);
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(const PartialIndexVector& outputIndices,
                                                            const DenseWeightVector<uint32>& weights) const override {
                return std::make_unique<StatisticsSubset<DenseWeightVector<uint32>, PartialIndexVector, uint32>>(
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
              const OutOfSampleWeightVector<DenseWeightVector<uint32>>& weights) const override {
                return std::make_unique<
                  StatisticsSubset<OutOfSampleWeightVector<DenseWeightVector<uint32>>, CompleteIndexVector, uint32>>(
                  *this->statePtr_, *this->ruleEvaluationFactory_, weights, outputIndices);
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(
              const PartialIndexVector& outputIndices,
              const OutOfSampleWeightVector<DenseWeightVector<uint32>>& weights) const override {
                return std::make_unique<
                  StatisticsSubset<OutOfSampleWeightVector<DenseWeightVector<uint32>>, PartialIndexVector, uint32>>(
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
              const DenseWeightVector<uint32>& weights) const override {
                return std::make_unique<WeightedStatistics<DenseWeightVector<uint32>, uint32>>(
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
            void visitScoreMatrix(IBoostingStatistics::DenseScoreMatrixVisitor denseVisitor,
                                  IBoostingStatistics::SparseScoreMatrixVisitor sparseVisitor) const override {
                sparseVisitor(this->statePtr_->scoreMatrixPtr->getView());
            }
    };

    template<typename OutputMatrix>
    static inline std::unique_ptr<IDecomposableStatistics<ISparseDecomposableRuleEvaluationFactory>> createStatistics(
      const ISparseDecomposableClassificationLossFactory& lossFactory,
      const ISparseEvaluationMeasureFactory& evaluationMeasureFactory,
      const ISparseDecomposableRuleEvaluationFactory& ruleEvaluationFactory,
      MultiThreadingSettings multiThreadingSettings, const OutputMatrix& outputMatrix) {
        uint32 numExamples = outputMatrix.numRows;
        uint32 numOutputs = outputMatrix.numCols;
        std::unique_ptr<ISparseDecomposableClassificationLoss> lossPtr =
          lossFactory.createSparseDecomposableClassificationLoss();
        std::unique_ptr<ISparseEvaluationMeasure> evaluationMeasurePtr =
          evaluationMeasureFactory.createSparseEvaluationMeasure();
        std::unique_ptr<SparseDecomposableStatisticMatrix> statisticMatrixPtr =
          std::make_unique<SparseDecomposableStatisticMatrix>(numExamples, numOutputs);
        std::unique_ptr<NumericSparseSetMatrix<float64>> scoreMatrixPtr =
          std::make_unique<NumericSparseSetMatrix<float64>>(numExamples, numOutputs);
        const ISparseDecomposableClassificationLoss* lossRawPtr = lossPtr.get();
        const OutputMatrix* outputMatrixPtr = &outputMatrix;
        const SparseSetView<float64>* scoreMatrixRawPtr = &scoreMatrixPtr->getView();
        SparseSetView<Statistic<float64>>* statisticMatrixRawPtr = &statisticMatrixPtr->getView();

#if MULTI_THREADING_SUPPORT_ENABLED
    #pragma omp parallel for firstprivate(numExamples) firstprivate(lossRawPtr) firstprivate(outputMatrixPtr) \
      firstprivate(scoreMatrixRawPtr) firstprivate(statisticMatrixRawPtr) schedule(dynamic) \
      num_threads(multiThreadingSettings.numThreads)
#endif
        for (int64 i = 0; i < numExamples; i++) {
            lossRawPtr->updateDecomposableStatistics(i, *outputMatrixPtr, *scoreMatrixRawPtr, IndexIterator(),
                                                     IndexIterator(outputMatrixPtr->numCols), *statisticMatrixRawPtr);
        }

        return std::make_unique<SparseDecomposableStatistics<OutputMatrix>>(
          std::move(lossPtr), std::move(evaluationMeasurePtr), ruleEvaluationFactory, outputMatrix,
          std::move(statisticMatrixPtr), std::move(scoreMatrixPtr));
    }

    SparseDecomposableClassificationStatisticsProviderFactory::
      SparseDecomposableClassificationStatisticsProviderFactory(
        std::unique_ptr<ISparseDecomposableClassificationLossFactory> lossFactoryPtr,
        std::unique_ptr<ISparseEvaluationMeasureFactory> evaluationMeasureFactoryPtr,
        std::unique_ptr<ISparseDecomposableRuleEvaluationFactory> regularRuleEvaluationFactoryPtr,
        std::unique_ptr<ISparseDecomposableRuleEvaluationFactory> pruningRuleEvaluationFactoryPtr,
        MultiThreadingSettings multiThreadingSettings)
        : lossFactoryPtr_(std::move(lossFactoryPtr)),
          evaluationMeasureFactoryPtr_(std::move(evaluationMeasureFactoryPtr)),
          regularRuleEvaluationFactoryPtr_(std::move(regularRuleEvaluationFactoryPtr)),
          pruningRuleEvaluationFactoryPtr_(std::move(pruningRuleEvaluationFactoryPtr)),
          multiThreadingSettings_(multiThreadingSettings) {}

    std::unique_ptr<IStatisticsProvider> SparseDecomposableClassificationStatisticsProviderFactory::create(
      const CContiguousView<const uint8>& labelMatrix) const {
        std::unique_ptr<IDecomposableStatistics<ISparseDecomposableRuleEvaluationFactory>> statisticsPtr =
          createStatistics(*lossFactoryPtr_, *evaluationMeasureFactoryPtr_, *regularRuleEvaluationFactoryPtr_,
                           multiThreadingSettings_, labelMatrix);
        return std::make_unique<DecomposableStatisticsProvider<ISparseDecomposableRuleEvaluationFactory>>(
          *regularRuleEvaluationFactoryPtr_, *pruningRuleEvaluationFactoryPtr_, std::move(statisticsPtr));
    }

    std::unique_ptr<IStatisticsProvider> SparseDecomposableClassificationStatisticsProviderFactory::create(
      const BinaryCsrView& labelMatrix) const {
        std::unique_ptr<IDecomposableStatistics<ISparseDecomposableRuleEvaluationFactory>> statisticsPtr =
          createStatistics(*lossFactoryPtr_, *evaluationMeasureFactoryPtr_, *regularRuleEvaluationFactoryPtr_,
                           multiThreadingSettings_, labelMatrix);
        return std::make_unique<DecomposableStatisticsProvider<ISparseDecomposableRuleEvaluationFactory>>(
          *regularRuleEvaluationFactoryPtr_, *pruningRuleEvaluationFactoryPtr_, std::move(statisticsPtr));
    }
}
