/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/boosting/losses/loss_decomposable.hpp"
#include "mlrl/common/iterator/binary_forward_iterator.hpp"
#include "mlrl/common/util/math.hpp"

#include <algorithm>

namespace boosting {

    /**
     * An implementation of the type `IDecomposableClassificationLoss` that relies on an "update function" and an
     * "evaluation function" for updating the gradients and Hessians and evaluating the predictions for an individual
     * label, respectively.
     */
    class DecomposableClassificationLoss : virtual public IDecomposableClassificationLoss {
        public:

            /**
             * A function that allows to update the gradient and Hessian for a single example and label. The function
             * accepts the true label, the predicted score, as well as references to the gradient and Hessian to be
             * updated, as arguments.
             */
            typedef void (*UpdateFunction)(bool trueLabel, float64 predictedScore, float64& gradient, float64& hessian);

            /**
             * A function that allows to calculate a numerical score that assesses the quality of the prediction for a
             * single example and label. The function accepts the true label and the predicted score as arguments and
             * returns a numerical score.
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
                                              CContiguousView<Tuple<float64>>& statisticView) const override final {
                CContiguousView<Tuple<float64>>::value_iterator statisticIterator =
                  statisticView.values_begin(exampleIndex);
                CContiguousView<float64>::value_const_iterator scoreIterator = scoreMatrix.values_cbegin(exampleIndex);
                CContiguousView<const uint8>::value_const_iterator labelIterator =
                  labelMatrix.values_cbegin(exampleIndex);
                uint32 numLabels = labelMatrix.numCols;

                for (uint32 i = 0; i < numLabels; i++) {
                    bool trueLabel = labelIterator[i];
                    float64 predictedScore = scoreIterator[i];
                    Tuple<float64>& tuple = statisticIterator[i];
                    (*updateFunction_)(trueLabel, predictedScore, tuple.first, tuple.second);
                }
            }

            void updateDecomposableStatistics(uint32 exampleIndex, const CContiguousView<const uint8>& labelMatrix,
                                              const CContiguousView<float64>& scoreMatrix,
                                              PartialIndexVector::const_iterator indicesBegin,
                                              PartialIndexVector::const_iterator indicesEnd,
                                              CContiguousView<Tuple<float64>>& statisticView) const override final {
                CContiguousView<Tuple<float64>>::value_iterator statisticIterator =
                  statisticView.values_begin(exampleIndex);
                CContiguousView<float64>::value_const_iterator scoreIterator = scoreMatrix.values_cbegin(exampleIndex);
                CContiguousView<const uint8>::value_const_iterator labelIterator =
                  labelMatrix.values_cbegin(exampleIndex);
                uint32 numLabels = indicesEnd - indicesBegin;

                for (uint32 i = 0; i < numLabels; i++) {
                    uint32 index = indicesBegin[i];
                    bool trueLabel = labelIterator[index];
                    float64 predictedScore = scoreIterator[index];
                    Tuple<float64>& tuple = statisticIterator[index];
                    (*updateFunction_)(trueLabel, predictedScore, tuple.first, tuple.second);
                }
            }

            void updateDecomposableStatistics(uint32 exampleIndex, const BinaryCsrView& labelMatrix,
                                              const CContiguousView<float64>& scoreMatrix,
                                              CompleteIndexVector::const_iterator indicesBegin,
                                              CompleteIndexVector::const_iterator indicesEnd,
                                              CContiguousView<Tuple<float64>>& statisticView) const override final {
                CContiguousView<Tuple<float64>>::value_iterator statisticIterator =
                  statisticView.values_begin(exampleIndex);
                CContiguousView<float64>::value_const_iterator scoreIterator = scoreMatrix.values_cbegin(exampleIndex);
                auto labelIterator = make_binary_forward_iterator(labelMatrix.indices_cbegin(exampleIndex),
                                                                  labelMatrix.indices_cend(exampleIndex));
                uint32 numLabels = labelMatrix.numCols;

                for (uint32 i = 0; i < numLabels; i++) {
                    bool trueLabel = *labelIterator;
                    float64 predictedScore = scoreIterator[i];
                    Tuple<float64>& tuple = statisticIterator[i];
                    (*updateFunction_)(trueLabel, predictedScore, tuple.first, tuple.second);
                    labelIterator++;
                }
            }

            void updateDecomposableStatistics(uint32 exampleIndex, const BinaryCsrView& labelMatrix,
                                              const CContiguousView<float64>& scoreMatrix,
                                              PartialIndexVector::const_iterator indicesBegin,
                                              PartialIndexVector::const_iterator indicesEnd,
                                              CContiguousView<Tuple<float64>>& statisticView) const override final {
                CContiguousView<Tuple<float64>>::value_iterator statisticIterator =
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
                    Tuple<float64>& tuple = statisticIterator[index];
                    (*updateFunction_)(trueLabel, predictedScore, tuple.first, tuple.second);
                }
            }

            /**
             * @see `IEvaluationMeasure::evaluate`
             */
            float64 evaluate(uint32 exampleIndex, const CContiguousView<const uint8>& labelMatrix,
                             const CContiguousView<float64>& scoreMatrix) const override final {
                CContiguousView<float64>::value_const_iterator scoreIterator = scoreMatrix.values_cbegin(exampleIndex);
                CContiguousView<const uint8>::value_const_iterator labelIterator =
                  labelMatrix.values_cbegin(exampleIndex);
                uint32 numLabels = labelMatrix.numCols;
                float64 mean = 0;

                for (uint32 i = 0; i < numLabels; i++) {
                    float64 predictedScore = scoreIterator[i];
                    bool trueLabel = labelIterator[i];
                    float64 score = (*evaluateFunction_)(trueLabel, predictedScore);
                    mean = iterativeArithmeticMean<float64>(i + 1, score, mean);
                }

                return mean;
            }

            /**
             * @see `IEvaluationMeasure::evaluate`
             */
            float64 evaluate(uint32 exampleIndex, const BinaryCsrView& labelMatrix,
                             const CContiguousView<float64>& scoreMatrix) const override final {
                CContiguousView<float64>::value_const_iterator scoreIterator = scoreMatrix.values_cbegin(exampleIndex);
                auto labelIterator = make_binary_forward_iterator(labelMatrix.indices_cbegin(exampleIndex),
                                                                  labelMatrix.indices_cend(exampleIndex));
                uint32 numLabels = labelMatrix.numCols;
                float64 mean = 0;

                for (uint32 i = 0; i < numLabels; i++) {
                    float64 predictedScore = scoreIterator[i];
                    bool trueLabel = *labelIterator;
                    float64 score = (*evaluateFunction_)(trueLabel, predictedScore);
                    mean = iterativeArithmeticMean<float64>(i + 1, score, mean);
                    labelIterator++;
                }

                return mean;
            }

            /**
             * @see `IDistanceMeasure::measureDistance`
             */
            float64 measureDistance(uint32 labelVectorIndex, const LabelVector& labelVector,
                                    View<float64>::const_iterator scoresBegin,
                                    View<float64>::const_iterator scoresEnd) const override final {
                uint32 numLabels = scoresEnd - scoresBegin;
                auto labelIterator = make_binary_forward_iterator(labelVector.cbegin(), labelVector.cend());
                float64 mean = 0;

                for (uint32 i = 0; i < numLabels; i++) {
                    float64 predictedScore = scoresBegin[i];
                    bool trueLabel = *labelIterator;
                    float64 score = (*evaluateFunction_)(trueLabel, predictedScore);
                    mean = iterativeArithmeticMean<float64>(i + 1, score, mean);
                    labelIterator++;
                }

                return mean;
            }
    };

}
