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

    template<typename GroundTruthIterator, typename UpdateFunction>
    static inline void updateDecomposableStatisticsInternally(View<Statistic<float64>>::iterator statisticIterator,
                                                              View<float64>::const_iterator scoreIterator,
                                                              GroundTruthIterator groundTruthIterator,
                                                              uint32 numOutputs, UpdateFunction updateFunction) {
        for (uint32 i = 0; i < numOutputs; i++) {
            typename util::iterator_value<GroundTruthIterator> groundTruth = *groundTruthIterator;
            float64 predictedScore = scoreIterator[i];
            Statistic<float64>& statistic = statisticIterator[i];
            (*updateFunction)(groundTruth, predictedScore, statistic.gradient, statistic.hessian);
            groundTruthIterator++;
        }
    }

    template<typename GroundTruthIterator, typename UpdateFunction>
    static inline void updateDecomposableStatisticsInternally(View<Statistic<float64>>::iterator statisticIterator,
                                                              View<float64>::const_iterator scoreIterator,
                                                              GroundTruthIterator groundTruthIterator,
                                                              PartialIndexVector::const_iterator indexIterator,
                                                              uint32 numOutputs, UpdateFunction updateFunction) {
        for (uint32 i = 0; i < numOutputs; i++) {
            uint32 index = indexIterator[i];
            typename util::iterator_value<GroundTruthIterator> groundTruth = groundTruthIterator[index];
            float64 predictedScore = scoreIterator[index];
            Statistic<float64>& statistic = statisticIterator[index];
            (*updateFunction)(groundTruth, predictedScore, statistic.gradient, statistic.hessian);
        }
    }

    template<typename GroundTruthIterator, typename EvaluateFunction>
    static inline float64 evaluateInternally(View<float64>::const_iterator scoreIterator,
                                             GroundTruthIterator groundTruthIterator, uint32 numOutputs,
                                             EvaluateFunction evaluateFunction) {
        float64 mean = 0;

        for (uint32 i = 0; i < numOutputs; i++) {
            float64 predictedScore = scoreIterator[i];
            typename util::iterator_value<GroundTruthIterator> groundTruth = *groundTruthIterator;
            float64 score = (*evaluateFunction)(groundTruth, predictedScore);
            mean = util::iterativeArithmeticMean<float64>(i + 1, score, mean);
            groundTruthIterator++;
        }

        return mean;
    }

    /**
     * An implementation of the type `IDecomposableClassificationLoss` that relies on an "update function" and an
     * "evaluation function" for updating the gradients and Hessians and evaluating the predictions for an individual
     * label, respectively.
     */
    class DecomposableClassificationLoss : virtual public IDecomposableClassificationLoss {
        public:

            /**
             * A function that allows to update the gradient and Hessian for a single example and label. The function
             * accepts the ground truth label, the predicted score, as well as references to the gradient and Hessian to
             * be updated, as arguments.
             */
            typedef void (*UpdateFunction)(bool trueLabel, float64 predictedScore, float64& gradient, float64& hessian);

            /**
             * A function that allows to calculate a numerical score that assesses the quality of the prediction for a
             * single example and label. The function accepts the ground truth label and the predicted score as
             * arguments and returns a numerical score.
             */
            typedef float64 (*EvaluateFunction)(bool trueLabel, float64 predictedScore);

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

            void updateDecomposableStatistics(uint32 exampleIndex, const CContiguousView<const uint8>& labelMatrix,
                                              const CContiguousView<float64>& scoreMatrix,
                                              CompleteIndexVector::const_iterator indicesBegin,
                                              CompleteIndexVector::const_iterator indicesEnd,
                                              CContiguousView<Statistic<float64>>& statisticView) const override final {
                updateDecomposableStatisticsInternally(
                  statisticView.values_begin(exampleIndex), scoreMatrix.values_cbegin(exampleIndex),
                  labelMatrix.values_cbegin(exampleIndex), labelMatrix.numCols, updateFunction_);
            }

            void updateDecomposableStatistics(uint32 exampleIndex, const CContiguousView<const uint8>& labelMatrix,
                                              const CContiguousView<float64>& scoreMatrix,
                                              PartialIndexVector::const_iterator indicesBegin,
                                              PartialIndexVector::const_iterator indicesEnd,
                                              CContiguousView<Statistic<float64>>& statisticView) const override final {
                uint32 numLabels = indicesEnd - indicesBegin;
                updateDecomposableStatisticsInternally(
                  statisticView.values_begin(exampleIndex), scoreMatrix.values_cbegin(exampleIndex),
                  labelMatrix.values_cbegin(exampleIndex), indicesBegin, numLabels, updateFunction_);
            }

            void updateDecomposableStatistics(uint32 exampleIndex, const BinaryCsrView& labelMatrix,
                                              const CContiguousView<float64>& scoreMatrix,
                                              CompleteIndexVector::const_iterator indicesBegin,
                                              CompleteIndexVector::const_iterator indicesEnd,
                                              CContiguousView<Statistic<float64>>& statisticView) const override final {
                updateDecomposableStatisticsInternally(
                  statisticView.values_begin(exampleIndex), scoreMatrix.values_cbegin(exampleIndex),
                  createBinarySparseForwardIterator(labelMatrix.indices_cbegin(exampleIndex),
                                                    labelMatrix.indices_cend(exampleIndex)),
                  labelMatrix.numCols, updateFunction_);
            }

            void updateDecomposableStatistics(uint32 exampleIndex, const BinaryCsrView& labelMatrix,
                                              const CContiguousView<float64>& scoreMatrix,
                                              PartialIndexVector::const_iterator indicesBegin,
                                              PartialIndexVector::const_iterator indicesEnd,
                                              CContiguousView<Statistic<float64>>& statisticView) const override final {
                CContiguousView<Statistic<float64>>::value_iterator statisticIterator =
                  statisticView.values_begin(exampleIndex);
                CContiguousView<float64>::value_const_iterator scoreIterator = scoreMatrix.values_cbegin(exampleIndex);
                BinaryCsrView::index_const_iterator labelIndicesBegin = labelMatrix.indices_cbegin(exampleIndex);
                BinaryCsrView::index_const_iterator labelIndicesEnd = labelMatrix.indices_cend(exampleIndex);
                uint32 numLabels = indicesEnd - indicesBegin;

                for (uint32 i = 0; i < numLabels; i++) {
                    uint32 index = indicesBegin[i];
                    labelIndicesBegin = std::lower_bound(labelIndicesBegin, labelIndicesEnd, index);
                    bool trueLabel = labelIndicesBegin != labelIndicesEnd && *labelIndicesBegin == index;
                    float64 predictedScore = scoreIterator[index];
                    Statistic<float64>& statistic = statisticIterator[index];
                    (*updateFunction_)(trueLabel, predictedScore, statistic.gradient, statistic.hessian);
                }
            }

            /**
             * @see `IClassificationEvaluationMeasure::evaluate`
             */
            float64 evaluate(uint32 exampleIndex, const CContiguousView<const uint8>& labelMatrix,
                             const CContiguousView<float64>& scoreMatrix) const override final {
                return evaluateInternally(scoreMatrix.values_cbegin(exampleIndex),
                                          labelMatrix.values_cbegin(exampleIndex), labelMatrix.numCols,
                                          evaluateFunction_);
            }

            /**
             * @see `IClassificationEvaluationMeasure::evaluate`
             */
            float64 evaluate(uint32 exampleIndex, const BinaryCsrView& labelMatrix,
                             const CContiguousView<float64>& scoreMatrix) const override final {
                return evaluateInternally(scoreMatrix.values_cbegin(exampleIndex),
                                          createBinarySparseForwardIterator(labelMatrix.indices_cbegin(exampleIndex),
                                                                            labelMatrix.indices_cend(exampleIndex)),
                                          labelMatrix.numCols, evaluateFunction_);
            }

            /**
             * @see `IDistanceMeasure::measureDistance`
             */
            float64 measureDistance(uint32 labelVectorIndex, const LabelVector& labelVector,
                                    View<float64>::const_iterator scoresBegin,
                                    View<float64>::const_iterator scoresEnd) const override final {
                uint32 numLabels = scoresEnd - scoresBegin;
                auto labelIterator = createBinarySparseForwardIterator(labelVector.cbegin(), labelVector.cend());
                float64 mean = 0;

                for (uint32 i = 0; i < numLabels; i++) {
                    float64 predictedScore = scoresBegin[i];
                    bool trueLabel = *labelIterator;
                    float64 score = (*evaluateFunction_)(trueLabel, predictedScore);
                    mean = util::iterativeArithmeticMean<float64>(i + 1, score, mean);
                    labelIterator++;
                }

                return mean;
            }
    };

    /**
     * An implementation of the type `IDecomposableRegressionLoss` that relies on an "update function" and an
     * "evaluation function" for updating the gradients and Hessians and evaluating the predictions for an individual
     * output, respectively.
     */
    class DecomposableRegressionLoss : virtual public IDecomposableRegressionLoss {
        public:

            /**
             * A function that allows to update the gradient and Hessian for a single example and output. The function
             * accepts the ground truth regression score, the predicted score, as well as references to the gradient and
             * Hessian to be updated, as arguments.
             */
            typedef void (*UpdateFunction)(float32 groundTruthScore, float64 predictedScore, float64& gradient,
                                           float64& hessian);

            /**
             * A function that allows to calculate a numerical score that assesses the quality of the prediction for a
             * single example and output. The function accepts the ground truth regression score and the predicted score
             * as arguments and returns a numerical score.
             */
            typedef float64 (*EvaluateFunction)(float32 groundTruthScore, float64 predictedScore);

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

            void updateDecomposableStatistics(uint32 exampleIndex,
                                              const CContiguousView<const float32>& regressionMatrix,
                                              const CContiguousView<float64>& scoreMatrix,
                                              CompleteIndexVector::const_iterator indicesBegin,
                                              CompleteIndexVector::const_iterator indicesEnd,
                                              CContiguousView<Statistic<float64>>& statisticView) const override final {
                updateDecomposableStatisticsInternally(
                  statisticView.values_begin(exampleIndex), scoreMatrix.values_cbegin(exampleIndex),
                  regressionMatrix.values_cbegin(exampleIndex), regressionMatrix.numCols, updateFunction_);
            }

            void updateDecomposableStatistics(uint32 exampleIndex,
                                              const CContiguousView<const float32>& regressionMatrix,
                                              const CContiguousView<float64>& scoreMatrix,
                                              PartialIndexVector::const_iterator indicesBegin,
                                              PartialIndexVector::const_iterator indicesEnd,
                                              CContiguousView<Statistic<float64>>& statisticView) const override final {
                uint32 numLabels = indicesEnd - indicesBegin;
                updateDecomposableStatisticsInternally(
                  statisticView.values_begin(exampleIndex), scoreMatrix.values_cbegin(exampleIndex),
                  regressionMatrix.values_cbegin(exampleIndex), numLabels, updateFunction_);
            }

            void updateDecomposableStatistics(uint32 exampleIndex, const CsrView<const float32>& regressionMatrix,
                                              const CContiguousView<float64>& scoreMatrix,
                                              CompleteIndexVector::const_iterator indicesBegin,
                                              CompleteIndexVector::const_iterator indicesEnd,
                                              CContiguousView<Statistic<float64>>& statisticView) const override final {
                updateDecomposableStatisticsInternally(
                  statisticView.values_begin(exampleIndex), scoreMatrix.values_cbegin(exampleIndex),
                  createSparseForwardIterator(
                    regressionMatrix.indices_cbegin(exampleIndex), regressionMatrix.indices_cend(exampleIndex),
                    regressionMatrix.values_cbegin(exampleIndex), regressionMatrix.values_cend(exampleIndex)),
                  regressionMatrix.numCols, updateFunction_);
            }

            void updateDecomposableStatistics(uint32 exampleIndex, const CsrView<const float32>& regressionMatrix,
                                              const CContiguousView<float64>& scoreMatrix,
                                              PartialIndexVector::const_iterator indicesBegin,
                                              PartialIndexVector::const_iterator indicesEnd,
                                              CContiguousView<Statistic<float64>>& statisticView) const override final {
                CContiguousView<Statistic<float64>>::value_iterator statisticIterator =
                  statisticView.values_begin(exampleIndex);
                CContiguousView<float64>::value_const_iterator scoreIterator = scoreMatrix.values_cbegin(exampleIndex);
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
                    float64 predictedScore = scoreIterator[index];
                    Statistic<float64>& statistic = statisticIterator[index];
                    (*updateFunction_)(groundTruth, predictedScore, statistic.gradient, statistic.hessian);
                }
            }

            /**
             * @see `IClassificationEvaluationMeasure::evaluate`
             */
            float64 evaluate(uint32 exampleIndex, const CContiguousView<const float32>& regressionMatrix,
                             const CContiguousView<float64>& scoreMatrix) const override final {
                return evaluateInternally(scoreMatrix.values_cbegin(exampleIndex),
                                          regressionMatrix.values_cbegin(exampleIndex), regressionMatrix.numCols,
                                          evaluateFunction_);
            }

            /**
             * @see `IClassificationEvaluationMeasure::evaluate`
             */
            float64 evaluate(uint32 exampleIndex, const CsrView<const float32>& regressionMatrix,
                             const CContiguousView<float64>& scoreMatrix) const override final {
                return evaluateInternally(scoreMatrix.values_cbegin(exampleIndex),
                                          createSparseForwardIterator(regressionMatrix.indices_cbegin(exampleIndex),
                                                                      regressionMatrix.indices_cend(exampleIndex),
                                                                      regressionMatrix.values_cbegin(exampleIndex),
                                                                      regressionMatrix.values_cend(exampleIndex)),
                                          regressionMatrix.numCols, evaluateFunction_);
            }
    };

}
