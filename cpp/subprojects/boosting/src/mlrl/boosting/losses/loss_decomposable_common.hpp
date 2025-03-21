/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/boosting/losses/loss_decomposable.hpp"
#include "mlrl/common/iterator/iterator_forward_sparse.hpp"
#include "mlrl/common/iterator/iterator_forward_sparse_binary.hpp"
#include "mlrl/common/util/iterators.hpp"
#include "mlrl/common/util/math.hpp"

#include <algorithm>

namespace boosting {

    template<typename StatisticIterator, typename ScoreIterator, typename GroundTruthIterator, typename UpdateFunction>
    static inline void updateDecomposableStatisticsInternally(StatisticIterator statisticIterator,
                                                              ScoreIterator scoreIterator,
                                                              GroundTruthIterator groundTruthIterator,
                                                              uint32 numOutputs, UpdateFunction updateFunction) {
        for (uint32 i = 0; i < numOutputs; i++) {
            typename util::iterator_value<GroundTruthIterator> groundTruth = *groundTruthIterator;
            typename util::iterator_value<ScoreIterator> predictedScore = scoreIterator[i];
            typename util::iterator_value<StatisticIterator>& statistic = statisticIterator[i];
            (*updateFunction)(groundTruth, predictedScore, statistic.gradient, statistic.hessian);
            groundTruthIterator++;
        }
    }

    template<typename StatisticIterator, typename ScoreIterator, typename GroundTruthIterator, typename UpdateFunction>
    static inline void updateDecomposableStatisticsInternally(StatisticIterator statisticIterator,
                                                              ScoreIterator scoreIterator,
                                                              GroundTruthIterator groundTruthIterator,
                                                              PartialIndexVector::const_iterator indexIterator,
                                                              uint32 numOutputs, UpdateFunction updateFunction) {
        for (uint32 i = 0; i < numOutputs; i++) {
            uint32 index = indexIterator[i];
            typename util::iterator_value<GroundTruthIterator> groundTruth = groundTruthIterator[index];
            typename util::iterator_value<ScoreIterator> predictedScore = scoreIterator[index];
            typename util::iterator_value<StatisticIterator>& statistic = statisticIterator[index];
            (*updateFunction)(groundTruth, predictedScore, statistic.gradient, statistic.hessian);
        }
    }

    template<typename ScoreIterator, typename GroundTruthIterator, typename EvaluateFunction>
    static inline typename util::iterator_value<ScoreIterator> evaluateInternally(
      ScoreIterator scoreIterator, GroundTruthIterator groundTruthIterator, uint32 numOutputs,
      EvaluateFunction evaluateFunction) {
        typedef util::iterator_value<ScoreIterator> score_type;
        score_type mean = 0;

        for (uint32 i = 0; i < numOutputs; i++) {
            score_type predictedScore = scoreIterator[i];
            typename util::iterator_value<GroundTruthIterator> groundTruth = *groundTruthIterator;
            score_type score = (*evaluateFunction)(groundTruth, predictedScore);
            mean = util::iterativeArithmeticMean(i + 1, score, mean);
            groundTruthIterator++;
        }

        return mean;
    }

    /**
     * An implementation of the type `IDecomposableClassificationLoss` that relies on an "update function" and an
     * "evaluation function" for updating the gradients and Hessians and evaluating the predictions for an individual
     * label, respectively.
     *
     * @tparam StatisticType The type of the gradients and Hessians that are calculated by the loss function
     */
    template<typename StatisticType>
    class DecomposableClassificationLoss : virtual public IDecomposableClassificationLoss<StatisticType> {
        public:

            /**
             * A function that allows to update the gradient and Hessian for a single example and label. The function
             * accepts the ground truth label, the predicted score, as well as references to the gradient and Hessian to
             * be updated, as arguments.
             */
            typedef void (*UpdateFunction)(bool trueLabel, StatisticType predictedScore, StatisticType& gradient,
                                           StatisticType& hessian);

            /**
             * A function that allows to calculate a numerical score that assesses the quality of the prediction for a
             * single example and label. The function accepts the ground truth label and the predicted score as
             * arguments and returns a numerical score.
             */
            typedef StatisticType (*EvaluateFunction)(bool trueLabel, StatisticType predictedScore);

            /**
             * The "update function" that is used for updating gradients and Hessians.
             */
            const UpdateFunction updateFunction_;

            /**
             * The "evaluation function" that is used for evaluating predictions.
             */
            const EvaluateFunction evaluateFunction_;

            /**
             * @param updateFunction    The "update function" to be used for updating gradients and Hessians
             * @param evaluateFunction  The "evaluation function" to be used for evaluating predictions
             */
            DecomposableClassificationLoss(UpdateFunction updateFunction, EvaluateFunction evaluateFunction)
                : updateFunction_(updateFunction), evaluateFunction_(evaluateFunction) {}

            void updateDecomposableStatistics(
              uint32 exampleIndex, const CContiguousView<const uint8>& labelMatrix,
              const CContiguousView<StatisticType>& scoreMatrix, CompleteIndexVector::const_iterator indicesBegin,
              CompleteIndexVector::const_iterator indicesEnd,
              CContiguousView<Statistic<StatisticType>>& statisticView) const override final {
                updateDecomposableStatisticsInternally(
                  statisticView.values_begin(exampleIndex), scoreMatrix.values_cbegin(exampleIndex),
                  labelMatrix.values_cbegin(exampleIndex), labelMatrix.numCols, updateFunction_);
            }

            void updateDecomposableStatistics(
              uint32 exampleIndex, const CContiguousView<const uint8>& labelMatrix,
              const CContiguousView<StatisticType>& scoreMatrix, PartialIndexVector::const_iterator indicesBegin,
              PartialIndexVector::const_iterator indicesEnd,
              CContiguousView<Statistic<StatisticType>>& statisticView) const override final {
                uint32 numLabels = indicesEnd - indicesBegin;
                updateDecomposableStatisticsInternally(
                  statisticView.values_begin(exampleIndex), scoreMatrix.values_cbegin(exampleIndex),
                  labelMatrix.values_cbegin(exampleIndex), indicesBegin, numLabels, updateFunction_);
            }

            void updateDecomposableStatistics(
              uint32 exampleIndex, const BinaryCsrView& labelMatrix, const CContiguousView<StatisticType>& scoreMatrix,
              CompleteIndexVector::const_iterator indicesBegin, CompleteIndexVector::const_iterator indicesEnd,
              CContiguousView<Statistic<StatisticType>>& statisticView) const override final {
                updateDecomposableStatisticsInternally(
                  statisticView.values_begin(exampleIndex), scoreMatrix.values_cbegin(exampleIndex),
                  createBinarySparseForwardIterator(labelMatrix.indices_cbegin(exampleIndex),
                                                    labelMatrix.indices_cend(exampleIndex)),
                  labelMatrix.numCols, updateFunction_);
            }

            void updateDecomposableStatistics(
              uint32 exampleIndex, const BinaryCsrView& labelMatrix, const CContiguousView<StatisticType>& scoreMatrix,
              PartialIndexVector::const_iterator indicesBegin, PartialIndexVector::const_iterator indicesEnd,
              CContiguousView<Statistic<StatisticType>>& statisticView) const override final {
                typename CContiguousView<Statistic<StatisticType>>::value_iterator statisticIterator =
                  statisticView.values_begin(exampleIndex);
                typename CContiguousView<StatisticType>::value_const_iterator scoreIterator =
                  scoreMatrix.values_cbegin(exampleIndex);
                BinaryCsrView::index_const_iterator labelIndicesBegin = labelMatrix.indices_cbegin(exampleIndex);
                BinaryCsrView::index_const_iterator labelIndicesEnd = labelMatrix.indices_cend(exampleIndex);
                uint32 numLabels = indicesEnd - indicesBegin;

                for (uint32 i = 0; i < numLabels; i++) {
                    uint32 index = indicesBegin[i];
                    labelIndicesBegin = std::lower_bound(labelIndicesBegin, labelIndicesEnd, index);
                    bool trueLabel = labelIndicesBegin != labelIndicesEnd && *labelIndicesBegin == index;
                    StatisticType predictedScore = scoreIterator[index];
                    Statistic<StatisticType>& statistic = statisticIterator[index];
                    (*updateFunction_)(trueLabel, predictedScore, statistic.gradient, statistic.hessian);
                }
            }

            /**
             * @see `IClassificationEvaluationMeasure::evaluate`
             */
            StatisticType evaluate(uint32 exampleIndex, const CContiguousView<const uint8>& labelMatrix,
                                   const CContiguousView<StatisticType>& scoreMatrix) const override final {
                return evaluateInternally(scoreMatrix.values_cbegin(exampleIndex),
                                          labelMatrix.values_cbegin(exampleIndex), labelMatrix.numCols,
                                          evaluateFunction_);
            }

            /**
             * @see `IClassificationEvaluationMeasure::evaluate`
             */
            StatisticType evaluate(uint32 exampleIndex, const BinaryCsrView& labelMatrix,
                                   const CContiguousView<StatisticType>& scoreMatrix) const override final {
                return evaluateInternally(scoreMatrix.values_cbegin(exampleIndex),
                                          createBinarySparseForwardIterator(labelMatrix.indices_cbegin(exampleIndex),
                                                                            labelMatrix.indices_cend(exampleIndex)),
                                          labelMatrix.numCols, evaluateFunction_);
            }

            /**
             * @see `IDistanceMeasure::measureDistance`
             */
            StatisticType measureDistance(uint32 labelVectorIndex, const LabelVector& labelVector,
                                          typename View<StatisticType>::const_iterator scoresBegin,
                                          typename View<StatisticType>::const_iterator scoresEnd) const override final {
                uint32 numLabels = scoresEnd - scoresBegin;
                auto labelIterator = createBinarySparseForwardIterator(labelVector.cbegin(), labelVector.cend());
                StatisticType mean = 0;

                for (uint32 i = 0; i < numLabels; i++) {
                    StatisticType predictedScore = scoresBegin[i];
                    bool trueLabel = *labelIterator;
                    StatisticType score = (*evaluateFunction_)(trueLabel, predictedScore);
                    mean = util::iterativeArithmeticMean(i + 1, score, mean);
                    labelIterator++;
                }

                return mean;
            }
    };

    /**
     * An implementation of the type `IDecomposableRegressionLoss` that relies on an "update function" and an
     * "evaluation function" for updating the gradients and Hessians and evaluating the predictions for an individual
     * output, respectively.
     *
     * @tparam StatisticType The type of the gradients and Hessians that are calculated by the loss function
     */
    template<typename StatisticType>
    class DecomposableRegressionLoss : virtual public IDecomposableRegressionLoss<StatisticType> {
        public:

            /**
             * A function that allows to update the gradient and Hessian for a single example and output. The function
             * accepts the ground truth regression score, the predicted score, as well as references to the gradient and
             * Hessian to be updated, as arguments.
             */
            typedef void (*UpdateFunction)(float32 groundTruthScore, StatisticType predictedScore,
                                           StatisticType& gradient, StatisticType& hessian);

            /**
             * A function that allows to calculate a numerical score that assesses the quality of the prediction for a
             * single example and output. The function accepts the ground truth regression score and the predicted score
             * as arguments and returns a numerical score.
             */
            typedef StatisticType (*EvaluateFunction)(float32 groundTruthScore, StatisticType predictedScore);

            /**
             * The "update function" that is used for updating gradients and Hessians.
             */
            const UpdateFunction updateFunction_;

            /**
             * The "evaluation function" that is used for evaluating predictions.
             */
            const EvaluateFunction evaluateFunction_;

            /**
             * @param updateFunction    The "update function" to be used for updating gradients and Hessians
             * @param evaluateFunction  The "evaluation function" to be used for evaluating predictions
             */
            DecomposableRegressionLoss(UpdateFunction updateFunction, EvaluateFunction evaluateFunction)
                : updateFunction_(updateFunction), evaluateFunction_(evaluateFunction) {}

            void updateDecomposableStatistics(
              uint32 exampleIndex, const CContiguousView<const float32>& regressionMatrix,
              const CContiguousView<StatisticType>& scoreMatrix, CompleteIndexVector::const_iterator indicesBegin,
              CompleteIndexVector::const_iterator indicesEnd,
              CContiguousView<Statistic<StatisticType>>& statisticView) const override final {
                updateDecomposableStatisticsInternally(
                  statisticView.values_begin(exampleIndex), scoreMatrix.values_cbegin(exampleIndex),
                  regressionMatrix.values_cbegin(exampleIndex), regressionMatrix.numCols, updateFunction_);
            }

            void updateDecomposableStatistics(
              uint32 exampleIndex, const CContiguousView<const float32>& regressionMatrix,
              const CContiguousView<StatisticType>& scoreMatrix, PartialIndexVector::const_iterator indicesBegin,
              PartialIndexVector::const_iterator indicesEnd,
              CContiguousView<Statistic<StatisticType>>& statisticView) const override final {
                uint32 numLabels = indicesEnd - indicesBegin;
                updateDecomposableStatisticsInternally(
                  statisticView.values_begin(exampleIndex), scoreMatrix.values_cbegin(exampleIndex),
                  regressionMatrix.values_cbegin(exampleIndex), numLabels, updateFunction_);
            }

            void updateDecomposableStatistics(
              uint32 exampleIndex, const CsrView<const float32>& regressionMatrix,
              const CContiguousView<StatisticType>& scoreMatrix, CompleteIndexVector::const_iterator indicesBegin,
              CompleteIndexVector::const_iterator indicesEnd,
              CContiguousView<Statistic<StatisticType>>& statisticView) const override final {
                updateDecomposableStatisticsInternally(
                  statisticView.values_begin(exampleIndex), scoreMatrix.values_cbegin(exampleIndex),
                  createSparseForwardIterator(
                    regressionMatrix.indices_cbegin(exampleIndex), regressionMatrix.indices_cend(exampleIndex),
                    regressionMatrix.values_cbegin(exampleIndex), regressionMatrix.values_cend(exampleIndex)),
                  regressionMatrix.numCols, updateFunction_);
            }

            void updateDecomposableStatistics(
              uint32 exampleIndex, const CsrView<const float32>& regressionMatrix,
              const CContiguousView<StatisticType>& scoreMatrix, PartialIndexVector::const_iterator indicesBegin,
              PartialIndexVector::const_iterator indicesEnd,
              CContiguousView<Statistic<StatisticType>>& statisticView) const override final {
                typename CContiguousView<Statistic<StatisticType>>::value_iterator statisticIterator =
                  statisticView.values_begin(exampleIndex);
                typename CContiguousView<StatisticType>::value_const_iterator scoreIterator =
                  scoreMatrix.values_cbegin(exampleIndex);
                CsrView<const float32>::value_const_iterator groundTruthValueIterator =
                  regressionMatrix.values_cbegin(exampleIndex);
                CsrView<const float32>::index_const_iterator groundTruthIndexIterator =
                  regressionMatrix.indices_cbegin(exampleIndex);
                CsrView<const float32>::index_const_iterator groundTruthIndicesBegin =
                  regressionMatrix.indices_cbegin(exampleIndex);
                CsrView<const float32>::index_const_iterator groundTruthIndicesEnd =
                  regressionMatrix.indices_cend(exampleIndex);
                uint32 numOutputs = indicesEnd - indicesBegin;

                for (uint32 i = 0; i < numOutputs; i++) {
                    uint32 index = indicesBegin[i];
                    groundTruthIndexIterator = std::lower_bound(groundTruthIndexIterator, groundTruthIndicesEnd, index);
                    uint32 offset = groundTruthIndexIterator - groundTruthIndicesBegin;
                    float32 groundTruth =
                      (groundTruthIndexIterator != groundTruthIndicesEnd && *groundTruthIndexIterator == index)
                        ? groundTruthValueIterator[offset]
                        : 0;
                    StatisticType predictedScore = scoreIterator[index];
                    Statistic<StatisticType>& statistic = statisticIterator[index];
                    (*updateFunction_)(groundTruth, predictedScore, statistic.gradient, statistic.hessian);
                }
            }

            /**
             * @see `IClassificationEvaluationMeasure::evaluate`
             */
            StatisticType evaluate(uint32 exampleIndex, const CContiguousView<const float32>& regressionMatrix,
                                   const CContiguousView<StatisticType>& scoreMatrix) const override final {
                return evaluateInternally(scoreMatrix.values_cbegin(exampleIndex),
                                          regressionMatrix.values_cbegin(exampleIndex), regressionMatrix.numCols,
                                          evaluateFunction_);
            }

            /**
             * @see `IClassificationEvaluationMeasure::evaluate`
             */
            StatisticType evaluate(uint32 exampleIndex, const CsrView<const float32>& regressionMatrix,
                                   const CContiguousView<StatisticType>& scoreMatrix) const override final {
                return evaluateInternally(scoreMatrix.values_cbegin(exampleIndex),
                                          createSparseForwardIterator(regressionMatrix.indices_cbegin(exampleIndex),
                                                                      regressionMatrix.indices_cend(exampleIndex),
                                                                      regressionMatrix.values_cbegin(exampleIndex),
                                                                      regressionMatrix.values_cend(exampleIndex)),
                                          regressionMatrix.numCols, evaluateFunction_);
            }
    };

}
