/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/boosting/data/matrix_c_contiguous_numeric.hpp"
#include "mlrl/boosting/data/vector_statistic_decomposable_dense.hpp"
#include "mlrl/boosting/losses/loss_decomposable.hpp"
#include "mlrl/common/measures/measure_evaluation.hpp"
#include "statistics_decomposable_common.hpp"

#include <memory>
#include <utility>

namespace boosting {

    /**
     * A matrix that stores gradients and Hessians that have been calculated using a decomposable loss function using
     * C-contiguous arrays.
     *
     * @tparam StatisticType The type of the gradients and Hessians
     */
    template<typename StatisticType>
    class DenseDecomposableStatisticMatrix final
        : public ClearableViewDecorator<MatrixDecorator<AllocatedCContiguousView<Statistic<StatisticType>>>> {
        public:

            /**
             * @param numRows   The number of rows in the matrix
             * @param numCols   The number of columns in the matrix
             */
            DenseDecomposableStatisticMatrix(uint32 numRows, uint32 numCols)
                : ClearableViewDecorator<MatrixDecorator<AllocatedCContiguousView<Statistic<StatisticType>>>>(
                    AllocatedCContiguousView<Statistic<StatisticType>>(numRows, numCols)) {}

            /**
             * Adds all gradients and Hessians in a vector to a specific row of this matrix. The gradients and Hessians
             * to be added are multiplied by a specific weight.
             *
             * @param row       The row
             * @param begin     An iterator to the beginning of the vector
             * @param end       An iterator to the end of the vector
             * @param weight    The weight, the gradients and Hessians should be multiplied by
             */
            void addToRow(uint32 row, typename View<Statistic<StatisticType>>::const_iterator begin,
                          typename View<Statistic<StatisticType>>::const_iterator end, uint32 weight) {
                util::addToViewWeighted(this->view.values_begin(row), begin, this->getNumCols(), weight);
            }
    };

    /**
     * Provides access to gradients and Hessians that have been calculated according to a decomposable loss function and
     * are stored using dense data structures.
     *
     * @tparam Loss                 The type of the loss function
     * @tparam OutputMatrix         The type of the matrix that provides access to the ground truth of the training
     *                              examples
     * @tparam EvaluationMeasure    The type of the evaluation that should be used to access the quality of predictions
     */
    template<typename Loss, typename OutputMatrix, typename EvaluationMeasure>
    class DenseDecomposableStatistics final
        : public AbstractDecomposableStatistics<OutputMatrix, DenseDecomposableStatisticMatrix<float64>,
                                                NumericCContiguousMatrix<float64>, Loss, EvaluationMeasure,
                                                IDecomposableRuleEvaluationFactory> {
        private:

            using StatisticsState = DecomposableStatisticsState<OutputMatrix, DenseDecomposableStatisticMatrix,
                                                                NumericCContiguousMatrix<float64>, Loss>;

            template<typename WeightVector, typename IndexVector>
            using StatisticsSubset = StatisticsSubset<StatisticsState, DenseDecomposableStatisticVector,
                                                      IDecomposableRuleEvaluationFactory, WeightVector, IndexVector>;

            template<typename WeightVector>
            using WeightedStatistics = WeightedStatistics<StatisticsState, DenseDecomposableStatisticVector,
                                                          IDecomposableRuleEvaluationFactory, WeightVector>;

        public:

            /**
             * @param lossPtr               An unique pointer to an object of template type `Loss` that implements the
             *                              loss function that should be used for calculating gradients and Hessians
             * @param evaluationMeasurePtr  An unique pointer to an object of template type `EvaluationMeasure` that
             *                              implements the evaluation measure that should be used to assess the quality
             *                              of predictions for a specific statistic
             * @param ruleEvaluationFactory A reference to an object of type `IDecomposableRuleEvaluationFactory`, that
             *                              allows to create instances of the class that is used for calculating the
             *                              predictions of rules, as well as their overall quality
             * @param outputMatrix          A reference to an object of template type `OutputMatrix` that provides
             *                              access to the ground truth of the training examples
             * @param statisticMatrixPtr    An unique pointer to an object of type `DenseDecomposableStatisticMatrix`
             *                              that provides access to the gradients and Hessians
             * @param scoreMatrixPtr        An unique pointer to an object of type `NumericCContiguousMatrix` that
             *                              stores the currently predicted scores
             */
            DenseDecomposableStatistics(std::unique_ptr<Loss> lossPtr,
                                        std::unique_ptr<EvaluationMeasure> evaluationMeasurePtr,
                                        const IDecomposableRuleEvaluationFactory& ruleEvaluationFactory,
                                        const OutputMatrix& outputMatrix,
                                        std::unique_ptr<DenseDecomposableStatisticMatrix<float64>> statisticMatrixPtr,
                                        std::unique_ptr<NumericCContiguousMatrix<float64>> scoreMatrixPtr)
                : AbstractDecomposableStatistics<OutputMatrix, DenseDecomposableStatisticMatrix<float64>,
                                                 NumericCContiguousMatrix<float64>, Loss, EvaluationMeasure,
                                                 IDecomposableRuleEvaluationFactory>(
                    std::move(lossPtr), std::move(evaluationMeasurePtr), ruleEvaluationFactory, outputMatrix,
                    std::move(statisticMatrixPtr), std::move(scoreMatrixPtr)) {}

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(const CompleteIndexVector& outputIndices,
                                                            const EqualWeightVector& weights) const override {
                return std::make_unique<StatisticsSubset<EqualWeightVector, CompleteIndexVector, float64>>(
                  *this->statePtr_, *this->ruleEvaluationFactory_, weights, outputIndices);
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(const PartialIndexVector& outputIndices,
                                                            const EqualWeightVector& weights) const override {
                return std::make_unique<StatisticsSubset<EqualWeightVector, PartialIndexVector, float64>>(
                  *this->statePtr_, *this->ruleEvaluationFactory_, weights, outputIndices);
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(const CompleteIndexVector& outputIndices,
                                                            const BitWeightVector& weights) const override {
                return std::make_unique<StatisticsSubset<BitWeightVector, CompleteIndexVector, float64>>(
                  *this->statePtr_, *this->ruleEvaluationFactory_, weights, outputIndices);
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(const PartialIndexVector& outputIndices,
                                                            const BitWeightVector& weights) const override {
                return std::make_unique<StatisticsSubset<BitWeightVector, PartialIndexVector, float64>>(
                  *this->statePtr_, *this->ruleEvaluationFactory_, weights, outputIndices);
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(const CompleteIndexVector& outputIndices,
                                                            const DenseWeightVector<uint32>& weights) const override {
                return std::make_unique<StatisticsSubset<DenseWeightVector<uint32>, CompleteIndexVector, float64>>(
                  *this->statePtr_, *this->ruleEvaluationFactory_, weights, outputIndices);
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(const PartialIndexVector& outputIndices,
                                                            const DenseWeightVector<uint32>& weights) const override {
                return std::make_unique<StatisticsSubset<DenseWeightVector<uint32>, PartialIndexVector, float64>>(
                  *this->statePtr_, *this->ruleEvaluationFactory_, weights, outputIndices);
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(
              const CompleteIndexVector& outputIndices,
              const DenseWeightVector<float32>& weights) const override final {
                return std::make_unique<StatisticsSubset<DenseWeightVector<float32>, CompleteIndexVector, float64>>(
                  *this->statePtr_, *this->ruleEvaluationFactory_, weights, outputIndices);
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(const PartialIndexVector& outputIndices,
                                                            const DenseWeightVector<float32>& weights) const override {
                return std::make_unique<StatisticsSubset<DenseWeightVector<float32>, PartialIndexVector, float64>>(
                  *this->statePtr_, *this->ruleEvaluationFactory_, weights, outputIndices);
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(
              const CompleteIndexVector& outputIndices,
              const OutOfSampleWeightVector<EqualWeightVector>& weights) const override {
                return std::make_unique<
                  StatisticsSubset<OutOfSampleWeightVector<EqualWeightVector>, CompleteIndexVector, float64>>(
                  *this->statePtr_, *this->ruleEvaluationFactory_, weights, outputIndices);
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(
              const PartialIndexVector& outputIndices,
              const OutOfSampleWeightVector<EqualWeightVector>& weights) const override {
                return std::make_unique<
                  StatisticsSubset<OutOfSampleWeightVector<EqualWeightVector>, PartialIndexVector, float64>>(
                  *this->statePtr_, *this->ruleEvaluationFactory_, weights, outputIndices);
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(
              const CompleteIndexVector& outputIndices,
              const OutOfSampleWeightVector<BitWeightVector>& weights) const override {
                return std::make_unique<
                  StatisticsSubset<OutOfSampleWeightVector<BitWeightVector>, CompleteIndexVector, float64>>(
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
                  StatisticsSubset<OutOfSampleWeightVector<DenseWeightVector<uint32>>, CompleteIndexVector, float64>>(
                  *this->statePtr_, *this->ruleEvaluationFactory_, weights, outputIndices);
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(
              const PartialIndexVector& outputIndices,
              const OutOfSampleWeightVector<DenseWeightVector<uint32>>& weights) const override {
                return std::make_unique<
                  StatisticsSubset<OutOfSampleWeightVector<DenseWeightVector<uint32>>, PartialIndexVector, float64>>(
                  *this->statePtr_, *this->ruleEvaluationFactory_, weights, outputIndices);
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(
              const CompleteIndexVector& outputIndices,
              const OutOfSampleWeightVector<DenseWeightVector<float32>>& weights) const override {
                return std::make_unique<
                  StatisticsSubset<OutOfSampleWeightVector<DenseWeightVector<float32>>, CompleteIndexVector, float64>>(
                  *this->statePtr_, *this->ruleEvaluationFactory_, weights, outputIndices);
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(
              const PartialIndexVector& outputIndices,
              const OutOfSampleWeightVector<DenseWeightVector<float32>>& weights) const override {
                return std::make_unique<
                  StatisticsSubset<OutOfSampleWeightVector<DenseWeightVector<float32>>, PartialIndexVector, float64>>(
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
                return std::make_unique<WeightedStatistics<DenseWeightVector<uint32>, float64>>(
                  *this->statePtr_, *this->ruleEvaluationFactory_, weights);
            }

            /**
             * @see `IStatistics::createWeightedStatistics`
             */
            std::unique_ptr<IWeightedStatistics> createWeightedStatistics(
              const DenseWeightVector<float32>& weights) const override {
                return std::make_unique<WeightedStatistics<DenseWeightVector<float32>, float64>>(
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
                dense64BitVisitor(this->statePtr_->scoreMatrixPtr->getView());
            }
    };

}
