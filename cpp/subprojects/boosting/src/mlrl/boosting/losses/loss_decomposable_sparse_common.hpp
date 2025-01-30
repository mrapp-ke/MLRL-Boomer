/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "loss_decomposable_common.hpp"
#include "mlrl/boosting/losses/loss_decomposable_sparse.hpp"
#include "mlrl/common/iterator/iterator_forward_non_zero_index.hpp"

#include <algorithm>
#include <limits>

namespace boosting {

    static inline constexpr uint32 LIMIT = std::numeric_limits<uint32>::max();

    template<typename StatisticType, typename IndexIterator>
    static inline uint32 fetchNextStatistic(
      IndexIterator& indexIterator, IndexIterator indicesEnd,
      typename SparseSetView<StatisticType>::value_const_iterator& scoreIterator,
      typename SparseSetView<StatisticType>::value_const_iterator scoresEnd, Statistic<StatisticType>& statistic,
      typename DecomposableClassificationLoss<StatisticType>::UpdateFunction updateFunction) {
        uint32 outputIndex = indexIterator == indicesEnd ? LIMIT : *indexIterator;
        uint32 scoreIndex = scoreIterator == scoresEnd ? LIMIT : (*scoreIterator).index;

        if (scoreIndex < outputIndex) {
            (*updateFunction)(false, (*scoreIterator).value, statistic.gradient, statistic.hessian);
            scoreIterator++;
            return scoreIndex;
        } else if (outputIndex < scoreIndex) {
            (*updateFunction)(true, 0, statistic.gradient, statistic.hessian);
            indexIterator++;
            return outputIndex;
        } else if (outputIndex < LIMIT) {
            (*updateFunction)(true, (*scoreIterator).value, statistic.gradient, statistic.hessian);
            scoreIterator++;
            indexIterator++;
            return outputIndex;
        }

        return LIMIT;
    }

    template<typename StatisticType, typename IndexIterator>
    static inline uint32 fetchNextNonZeroStatistic(
      IndexIterator& indexIterator, IndexIterator indicesEnd,
      typename SparseSetView<StatisticType>::value_const_iterator& scoreIterator,
      typename SparseSetView<StatisticType>::value_const_iterator scoresEnd, Statistic<StatisticType>& statistic,
      typename DecomposableClassificationLoss<StatisticType>::UpdateFunction updateFunction) {
        uint32 index =
          fetchNextStatistic(indexIterator, indicesEnd, scoreIterator, scoresEnd, statistic, updateFunction);

        while (index < LIMIT && isEqualToZero(statistic.gradient)) {
            index = fetchNextStatistic(indexIterator, indicesEnd, scoreIterator, scoresEnd, statistic, updateFunction);
        }

        return index;
    }

    template<typename StatisticType, typename IndexIterator>
    static inline void updateDecomposableStatisticsInternally(
      IndexIterator indicesBegin, IndexIterator indicesEnd,
      typename SparseSetView<StatisticType>::value_const_iterator scoresBegin,
      typename SparseSetView<StatisticType>::value_const_iterator scoresEnd,
      typename SparseSetView<Statistic<StatisticType>>::row row,
      typename DecomposableClassificationLoss<StatisticType>::UpdateFunction updateFunction) {
        row.clear();
        Statistic<StatisticType> statistic;
        uint32 index;

        while ((index = fetchNextNonZeroStatistic(indicesBegin, indicesEnd, scoresBegin, scoresEnd, statistic,
                                                  updateFunction))
               < LIMIT) {
            IndexedValue<Statistic<StatisticType>>& entry = row.emplace(index);
            entry.value = statistic;
        }
    }

    template<typename ScoreType, typename IndexIterator>
    static inline uint32 fetchNextEvaluation(
      IndexIterator& indexIterator, IndexIterator indicesEnd,
      typename SparseSetView<ScoreType>::value_const_iterator& scoreIterator,
      typename SparseSetView<ScoreType>::value_const_iterator scoresEnd, ScoreType& score,
      typename DecomposableClassificationLoss<ScoreType>::EvaluateFunction evaluateFunction) {
        uint32 outputIndex = indexIterator == indicesEnd ? LIMIT : *indexIterator;
        uint32 scoreIndex = scoreIterator == scoresEnd ? LIMIT : (*scoreIterator).index;

        if (scoreIndex < outputIndex) {
            score = (*evaluateFunction)(false, (*scoreIterator).value);
            scoreIterator++;
            return scoreIndex;
        } else if (outputIndex < scoreIndex) {
            score = (*evaluateFunction)(true, 0);
            indexIterator++;
            return outputIndex;
        } else if (outputIndex < LIMIT) {
            score = (*evaluateFunction)(true, (*scoreIterator).value);
            scoreIterator++;
            indexIterator++;
            return outputIndex;
        }

        return LIMIT;
    }

    template<typename ScoreType, typename IndexIterator>
    static inline uint32 fetchNextNonZeroEvaluation(
      IndexIterator& indexIterator, IndexIterator indicesEnd,
      typename SparseSetView<ScoreType>::value_const_iterator& scoreIterator,
      typename SparseSetView<ScoreType>::value_const_iterator scoresEnd, ScoreType& score,
      typename DecomposableClassificationLoss<ScoreType>::EvaluateFunction evaluateFunction) {
        uint32 index =
          fetchNextEvaluation(indexIterator, indicesEnd, scoreIterator, scoresEnd, score, evaluateFunction);

        while (index < LIMIT && isEqualToZero(score)) {
            index = fetchNextEvaluation(indexIterator, indicesEnd, scoreIterator, scoresEnd, score, evaluateFunction);
        }

        return index;
    }

    template<typename ScoreType, typename IndexIterator>
    static inline ScoreType evaluateInternally(
      IndexIterator indicesBegin, IndexIterator indicesEnd,
      typename SparseSetView<ScoreType>::value_const_iterator scoresBegin,
      typename SparseSetView<ScoreType>::value_const_iterator scoresEnd,
      typename DecomposableClassificationLoss<ScoreType>::EvaluateFunction evaluateFunction, uint32 numLabels) {
        ScoreType mean = 0;
        ScoreType score = 0;
        uint32 i = 0;

        while (fetchNextNonZeroEvaluation(indicesBegin, indicesEnd, scoresBegin, scoresEnd, score, evaluateFunction)
               < LIMIT) {
            mean = util::iterativeArithmeticMean(i + 1, score, mean);
            i++;
        }

        return mean * ((ScoreType) i / (ScoreType) numLabels);
    }

    /**
     * An implementation of the type `ISparseDecomposableClassificationLoss` that relies on an "update function" and an
     * "evaluation function" for updating the gradients and Hessians and evaluation the predictions for an individual
     * label, respectively.
     */
    template<typename StatisticType>
    class SparseDecomposableClassificationLoss final : public DecomposableClassificationLoss<StatisticType>,
                                                       public ISparseDecomposableClassificationLoss<StatisticType> {
        public:

            /**
             * @param updateFunction    The "update function" to be used for updating gradients and Hessians
             * @param evaluateFunction  The "evaluation function" to be used for evaluating predictions
             */
            SparseDecomposableClassificationLoss(
              typename DecomposableClassificationLoss<StatisticType>::UpdateFunction updateFunction,
              typename DecomposableClassificationLoss<StatisticType>::EvaluateFunction evaluateFunction)
                : DecomposableClassificationLoss<StatisticType>(updateFunction, evaluateFunction) {}

            // Keep functions from the parent class rather than hiding them.
            using DecomposableClassificationLoss<StatisticType>::evaluate;
            using DecomposableClassificationLoss<StatisticType>::updateDecomposableStatistics;

            void updateDecomposableStatistics(uint32 exampleIndex, const CContiguousView<const uint8>& labelMatrix,
                                              const SparseSetView<StatisticType>& scoreMatrix,
                                              CompleteIndexVector::const_iterator indicesBegin,
                                              CompleteIndexVector::const_iterator indicesEnd,
                                              SparseSetView<Statistic<StatisticType>>& statisticView) const override {
                auto labelIndicesBegin = createNonZeroIndexForwardIterator(labelMatrix.values_cbegin(exampleIndex),
                                                                           labelMatrix.values_cend(exampleIndex));
                auto labelIndicesEnd = createNonZeroIndexForwardIterator(labelMatrix.values_cend(exampleIndex),
                                                                         labelMatrix.values_cend(exampleIndex));
                updateDecomposableStatisticsInternally<StatisticType, decltype(labelIndicesBegin)>(
                  labelIndicesBegin, labelIndicesEnd, scoreMatrix.values_cbegin(exampleIndex),
                  scoreMatrix.values_cend(exampleIndex), statisticView[exampleIndex], this->updateFunction_);
            }

            void updateDecomposableStatistics(uint32 exampleIndex, const CContiguousView<const uint8>& labelMatrix,
                                              const SparseSetView<StatisticType>& scoreMatrix,
                                              PartialIndexVector::const_iterator indicesBegin,
                                              PartialIndexVector::const_iterator indicesEnd,
                                              SparseSetView<Statistic<StatisticType>>& statisticView) const override {
                const typename SparseSetView<StatisticType>::const_row scoreMatrixRow = scoreMatrix[exampleIndex];
                CContiguousView<const uint8>::value_const_iterator labelIterator =
                  labelMatrix.values_cbegin(exampleIndex);
                typename SparseSetView<Statistic<StatisticType>>::row statisticViewRow = statisticView[exampleIndex];
                uint32 numElements = indicesEnd - indicesBegin;
                Statistic<StatisticType> statistic;

                for (uint32 i = 0; i < numElements; i++) {
                    uint32 index = indicesBegin[i];
                    const IndexedValue<StatisticType>* scoreMatrixEntry = scoreMatrixRow[index];
                    StatisticType predictedScore = scoreMatrixEntry ? scoreMatrixEntry->value : 0;
                    bool trueLabel = labelIterator[index];
                    (*this->updateFunction_)(trueLabel, predictedScore, statistic.gradient, statistic.hessian);

                    if (!isEqualToZero(statistic.gradient)) {
                        IndexedValue<Statistic<StatisticType>>& statisticViewEntry = statisticViewRow.emplace(index);
                        statisticViewEntry.value = statistic;
                    } else {
                        statisticViewRow.erase(index);
                    }
                }
            }

            void updateDecomposableStatistics(uint32 exampleIndex, const BinaryCsrView& labelMatrix,
                                              const SparseSetView<StatisticType>& scoreMatrix,
                                              CompleteIndexVector::const_iterator indicesBegin,
                                              CompleteIndexVector::const_iterator indicesEnd,
                                              SparseSetView<Statistic<StatisticType>>& statisticView) const override {
                updateDecomposableStatisticsInternally<StatisticType, BinaryCsrView::index_const_iterator>(
                  labelMatrix.indices_cbegin(exampleIndex), labelMatrix.indices_cend(exampleIndex),
                  scoreMatrix.values_cbegin(exampleIndex), scoreMatrix.values_cend(exampleIndex),
                  statisticView[exampleIndex], this->updateFunction_);
            }

            void updateDecomposableStatistics(uint32 exampleIndex, const BinaryCsrView& labelMatrix,
                                              const SparseSetView<StatisticType>& scoreMatrix,
                                              PartialIndexVector::const_iterator indicesBegin,
                                              PartialIndexVector::const_iterator indicesEnd,
                                              SparseSetView<Statistic<StatisticType>>& statisticView) const override {
                const typename SparseSetView<StatisticType>::const_row scoreMatrixRow = scoreMatrix[exampleIndex];
                BinaryCsrView::index_const_iterator labelIndicesBegin = labelMatrix.indices_cbegin(exampleIndex);
                BinaryCsrView::index_const_iterator labelIndicesEnd = labelMatrix.indices_cend(exampleIndex);
                typename SparseSetView<Statistic<StatisticType>>::row statisticViewRow = statisticView[exampleIndex];
                uint32 numElements = indicesEnd - indicesBegin;
                Statistic<StatisticType> statistic;

                for (uint32 i = 0; i < numElements; i++) {
                    uint32 index = indicesBegin[i];
                    labelIndicesBegin = std::lower_bound(labelIndicesBegin, labelIndicesEnd, index);
                    bool trueLabel = labelIndicesBegin != labelIndicesEnd && *labelIndicesBegin == index;
                    const IndexedValue<StatisticType>* scoreMatrixEntry = scoreMatrixRow[index];
                    StatisticType predictedScore = scoreMatrixEntry ? scoreMatrixEntry->value : 0;
                    (*this->updateFunction_)(trueLabel, predictedScore, statistic.gradient, statistic.hessian);

                    if (!isEqualToZero(statistic.gradient)) {
                        IndexedValue<Statistic<StatisticType>>& statisticViewEntry = statisticViewRow.emplace(index);
                        statisticViewEntry.value = statistic;
                    } else {
                        statisticViewRow.erase(index);
                    }
                }
            }

            /**
             * @see `IClassificationEvaluationMeasure::evaluate`
             */
            StatisticType evaluate(uint32 exampleIndex, const CContiguousView<const uint8>& labelMatrix,
                                   const SparseSetView<StatisticType>& scoreMatrix) const override {
                auto indicesBegin = createNonZeroIndexForwardIterator(labelMatrix.values_cbegin(exampleIndex),
                                                                      labelMatrix.values_cend(exampleIndex));
                auto indicesEnd = createNonZeroIndexForwardIterator(labelMatrix.values_cend(exampleIndex),
                                                                    labelMatrix.values_cend(exampleIndex));
                return evaluateInternally<StatisticType, decltype(indicesBegin)>(
                  indicesBegin, indicesEnd, scoreMatrix.values_cbegin(exampleIndex),
                  scoreMatrix.values_cend(exampleIndex), this->evaluateFunction_, labelMatrix.numCols);
            }

            /**
             * @see `IClassificationEvaluationMeasure::evaluate`
             */
            StatisticType evaluate(uint32 exampleIndex, const BinaryCsrView& labelMatrix,
                                   const SparseSetView<StatisticType>& scoreMatrix) const override {
                return evaluateInternally<StatisticType, BinaryCsrView::index_const_iterator>(
                  labelMatrix.indices_cbegin(exampleIndex), labelMatrix.indices_cend(exampleIndex),
                  scoreMatrix.values_cbegin(exampleIndex), scoreMatrix.values_cend(exampleIndex),
                  this->evaluateFunction_, labelMatrix.numCols);
            }
    };

}
