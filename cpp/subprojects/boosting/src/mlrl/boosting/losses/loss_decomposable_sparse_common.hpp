/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#ifdef _WIN32
    #pragma warning(push)
    #pragma warning(disable : 4250)
#endif

#include "loss_decomposable_common.hpp"
#include "mlrl/boosting/losses/loss_decomposable_sparse.hpp"
#include "mlrl/common/iterator/non_zero_index_forward_iterator.hpp"

#include <algorithm>
#include <limits>

namespace boosting {

    static const uint32 LIMIT = std::numeric_limits<uint32>::max();

    template<typename IndexIterator>
    static inline uint32 fetchNextStatistic(IndexIterator& indexIterator, IndexIterator indicesEnd,
                                            SparseSetView<float64>::value_const_iterator& scoreIterator,
                                            SparseSetView<float64>::value_const_iterator scoresEnd,
                                            Tuple<float64>& tuple,
                                            DecomposableClassificationLoss::UpdateFunction updateFunction) {
        uint32 outputIndex = indexIterator == indicesEnd ? LIMIT : *indexIterator;
        uint32 scoreIndex = scoreIterator == scoresEnd ? LIMIT : (*scoreIterator).index;

        if (scoreIndex < outputIndex) {
            (*updateFunction)(false, (*scoreIterator).value, tuple.first, tuple.second);
            scoreIterator++;
            return scoreIndex;
        } else if (outputIndex < scoreIndex) {
            (*updateFunction)(true, 0, tuple.first, tuple.second);
            indexIterator++;
            return outputIndex;
        } else if (outputIndex < LIMIT) {
            (*updateFunction)(true, (*scoreIterator).value, tuple.first, tuple.second);
            scoreIterator++;
            indexIterator++;
            return outputIndex;
        }

        return LIMIT;
    }

    template<typename IndexIterator>
    static inline uint32 fetchNextNonZeroStatistic(IndexIterator& indexIterator, IndexIterator indicesEnd,
                                                   SparseSetView<float64>::value_const_iterator& scoreIterator,
                                                   SparseSetView<float64>::value_const_iterator scoresEnd,
                                                   Tuple<float64>& tuple,
                                                   DecomposableClassificationLoss::UpdateFunction updateFunction) {
        uint32 index = fetchNextStatistic(indexIterator, indicesEnd, scoreIterator, scoresEnd, tuple, updateFunction);

        while (index < LIMIT && isEqualToZero(tuple.first)) {
            index = fetchNextStatistic(indexIterator, indicesEnd, scoreIterator, scoresEnd, tuple, updateFunction);
        }

        return index;
    }

    template<typename IndexIterator>
    static inline void updateDecomposableStatisticsInternally(
      IndexIterator indicesBegin, IndexIterator indicesEnd, SparseSetView<float64>::value_const_iterator scoresBegin,
      SparseSetView<float64>::value_const_iterator scoresEnd, SparseSetView<Tuple<float64>>::row row,
      DecomposableClassificationLoss::UpdateFunction updateFunction) {
        row.clear();
        Tuple<float64> tuple;
        uint32 index;

        while (
          (index = fetchNextNonZeroStatistic(indicesBegin, indicesEnd, scoresBegin, scoresEnd, tuple, updateFunction))
          < LIMIT) {
            IndexedValue<Tuple<float64>>& entry = row.emplace(index);
            entry.value = tuple;
        }
    }

    template<typename IndexIterator>
    static inline uint32 fetchNextEvaluation(IndexIterator& indexIterator, IndexIterator indicesEnd,
                                             SparseSetView<float64>::value_const_iterator& scoreIterator,
                                             SparseSetView<float64>::value_const_iterator scoresEnd, float64& score,
                                             DecomposableClassificationLoss::EvaluateFunction evaluateFunction) {
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

    template<typename IndexIterator>
    static inline uint32 fetchNextNonZeroEvaluation(IndexIterator& indexIterator, IndexIterator indicesEnd,
                                                    SparseSetView<float64>::value_const_iterator& scoreIterator,
                                                    SparseSetView<float64>::value_const_iterator scoresEnd,
                                                    float64& score,
                                                    DecomposableClassificationLoss::EvaluateFunction evaluateFunction) {
        uint32 index =
          fetchNextEvaluation(indexIterator, indicesEnd, scoreIterator, scoresEnd, score, evaluateFunction);

        while (index < LIMIT && isEqualToZero(score)) {
            index = fetchNextEvaluation(indexIterator, indicesEnd, scoreIterator, scoresEnd, score, evaluateFunction);
        }

        return index;
    }

    template<typename IndexIterator>
    static inline float64 evaluateInternally(IndexIterator indicesBegin, IndexIterator indicesEnd,
                                             SparseSetView<float64>::value_const_iterator scoresBegin,
                                             SparseSetView<float64>::value_const_iterator scoresEnd,
                                             DecomposableClassificationLoss::EvaluateFunction evaluateFunction,
                                             uint32 numLabels) {
        float64 mean = 0;
        float64 score = 0;
        uint32 i = 0;

        while (fetchNextNonZeroEvaluation(indicesBegin, indicesEnd, scoresBegin, scoresEnd, score, evaluateFunction)
               < LIMIT) {
            mean = iterativeArithmeticMean<float64>(i + 1, score, mean);
            i++;
        }

        return mean * ((float64) i / (float64) numLabels);
    }

    /**
     * An implementation of the type `ISparseDecomposableClassificationLoss` that relies on an "update function" and an
     * "evaluation function" for updating the gradients and Hessians and evaluation the predictions for an individual
     * label, respectively.
     */
    class SparseDecomposableClassificationLoss final : public DecomposableClassificationLoss,
                                                       public ISparseDecomposableClassificationLoss {
        public:

            /**
             * @param updateFunction    The "update function" to be used for updating gradients and Hessians
             * @param evaluateFunction  The "evaluation function" to be used for evaluating predictions
             */
            SparseDecomposableClassificationLoss(UpdateFunction updateFunction, EvaluateFunction evaluateFunction)
                : DecomposableClassificationLoss(updateFunction, evaluateFunction) {}

            // Keep functions from the parent class rather than hiding them.
            using DecomposableClassificationLoss::evaluate;
            using DecomposableClassificationLoss::updateDecomposableStatistics;

            void updateDecomposableStatistics(uint32 exampleIndex, const CContiguousView<const uint8>& labelMatrix,
                                              const SparseSetView<float64>& scoreMatrix,
                                              CompleteIndexVector::const_iterator indicesBegin,
                                              CompleteIndexVector::const_iterator indicesEnd,
                                              SparseSetView<Tuple<float64>>& statisticView) const override {
                auto labelIndicesBegin = make_non_zero_index_forward_iterator(labelMatrix.values_cbegin(exampleIndex),
                                                                              labelMatrix.values_cend(exampleIndex));
                auto labelIndicesEnd = make_non_zero_index_forward_iterator(labelMatrix.values_cend(exampleIndex),
                                                                            labelMatrix.values_cend(exampleIndex));
                updateDecomposableStatisticsInternally(
                  labelIndicesBegin, labelIndicesEnd, scoreMatrix.values_cbegin(exampleIndex),
                  scoreMatrix.values_cend(exampleIndex), statisticView[exampleIndex],
                  DecomposableClassificationLoss::updateFunction_);
            }

            void updateDecomposableStatistics(uint32 exampleIndex, const CContiguousView<const uint8>& labelMatrix,
                                              const SparseSetView<float64>& scoreMatrix,
                                              PartialIndexVector::const_iterator indicesBegin,
                                              PartialIndexVector::const_iterator indicesEnd,
                                              SparseSetView<Tuple<float64>>& statisticView) const override {
                const SparseSetView<float64>::const_row scoreMatrixRow = scoreMatrix[exampleIndex];
                CContiguousView<const uint8>::value_const_iterator labelIterator =
                  labelMatrix.values_cbegin(exampleIndex);
                SparseSetView<Tuple<float64>>::row statisticViewRow = statisticView[exampleIndex];
                uint32 numElements = indicesEnd - indicesBegin;
                Tuple<float64> tuple;

                for (uint32 i = 0; i < numElements; i++) {
                    uint32 index = indicesBegin[i];
                    const IndexedValue<float64>* scoreMatrixEntry = scoreMatrixRow[index];
                    float64 predictedScore = scoreMatrixEntry ? scoreMatrixEntry->value : 0;
                    bool trueLabel = labelIterator[index];
                    (*DecomposableClassificationLoss::updateFunction_)(trueLabel, predictedScore, tuple.first,
                                                                       tuple.second);

                    if (!isEqualToZero(tuple.first)) {
                        IndexedValue<Tuple<float64>>& statisticViewEntry = statisticViewRow.emplace(index);
                        statisticViewEntry.value = tuple;
                    } else {
                        statisticViewRow.erase(index);
                    }
                }
            }

            void updateDecomposableStatistics(uint32 exampleIndex, const BinaryCsrView& labelMatrix,
                                              const SparseSetView<float64>& scoreMatrix,
                                              CompleteIndexVector::const_iterator indicesBegin,
                                              CompleteIndexVector::const_iterator indicesEnd,
                                              SparseSetView<Tuple<float64>>& statisticView) const override {
                updateDecomposableStatisticsInternally(
                  labelMatrix.indices_cbegin(exampleIndex), labelMatrix.indices_cend(exampleIndex),
                  scoreMatrix.values_cbegin(exampleIndex), scoreMatrix.values_cend(exampleIndex),
                  statisticView[exampleIndex], DecomposableClassificationLoss::updateFunction_);
            }

            void updateDecomposableStatistics(uint32 exampleIndex, const BinaryCsrView& labelMatrix,
                                              const SparseSetView<float64>& scoreMatrix,
                                              PartialIndexVector::const_iterator indicesBegin,
                                              PartialIndexVector::const_iterator indicesEnd,
                                              SparseSetView<Tuple<float64>>& statisticView) const override {
                const SparseSetView<float64>::const_row scoreMatrixRow = scoreMatrix[exampleIndex];
                BinaryCsrView::index_const_iterator labelIndicesBegin = labelMatrix.indices_cbegin(exampleIndex);
                BinaryCsrView::index_const_iterator labelIndicesEnd = labelMatrix.indices_cend(exampleIndex);
                SparseSetView<Tuple<float64>>::row statisticViewRow = statisticView[exampleIndex];
                uint32 numElements = indicesEnd - indicesBegin;
                Tuple<float64> tuple;

                for (uint32 i = 0; i < numElements; i++) {
                    uint32 index = indicesBegin[i];
                    labelIndicesBegin = std::lower_bound(labelIndicesBegin, labelIndicesEnd, index);
                    bool trueLabel = labelIndicesBegin != labelIndicesEnd && *labelIndicesBegin == index;
                    const IndexedValue<float64>* scoreMatrixEntry = scoreMatrixRow[index];
                    float64 predictedScore = scoreMatrixEntry ? scoreMatrixEntry->value : 0;
                    (*DecomposableClassificationLoss::updateFunction_)(trueLabel, predictedScore, tuple.first,
                                                                       tuple.second);

                    if (!isEqualToZero(tuple.first)) {
                        IndexedValue<Tuple<float64>>& statisticViewEntry = statisticViewRow.emplace(index);
                        statisticViewEntry.value = tuple;
                    } else {
                        statisticViewRow.erase(index);
                    }
                }
            }

            /**
             * @see `IClassificationEvaluationMeasure::evaluate`
             */
            float64 evaluate(uint32 exampleIndex, const CContiguousView<const uint8>& labelMatrix,
                             const SparseSetView<float64>& scoreMatrix) const override {
                auto indicesBegin = make_non_zero_index_forward_iterator(labelMatrix.values_cbegin(exampleIndex),
                                                                         labelMatrix.values_cend(exampleIndex));
                auto indicesEnd = make_non_zero_index_forward_iterator(labelMatrix.values_cend(exampleIndex),
                                                                       labelMatrix.values_cend(exampleIndex));
                return evaluateInternally(indicesBegin, indicesEnd, scoreMatrix.values_cbegin(exampleIndex),
                                          scoreMatrix.values_cend(exampleIndex),
                                          DecomposableClassificationLoss::evaluateFunction_, labelMatrix.numCols);
            }

            /**
             * @see `IClassificationEvaluationMeasure::evaluate`
             */
            float64 evaluate(uint32 exampleIndex, const BinaryCsrView& labelMatrix,
                             const SparseSetView<float64>& scoreMatrix) const override {
                return evaluateInternally(
                  labelMatrix.indices_cbegin(exampleIndex), labelMatrix.indices_cend(exampleIndex),
                  scoreMatrix.values_cbegin(exampleIndex), scoreMatrix.values_cend(exampleIndex),
                  DecomposableClassificationLoss::evaluateFunction_, labelMatrix.numCols);
            }
    };

}

#ifdef _WIN32
    #pragma warning(pop)
#endif
