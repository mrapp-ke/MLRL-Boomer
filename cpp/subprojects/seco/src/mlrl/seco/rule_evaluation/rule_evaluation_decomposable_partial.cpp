#include "mlrl/seco/rule_evaluation/rule_evaluation_decomposable_partial.hpp"

#include "mlrl/common/data/vector_sparse_array.hpp"
#include "mlrl/common/indices/index_vector_partial.hpp"
#include "mlrl/common/iterator/iterator_forward_sparse_binary.hpp"
#include "mlrl/common/rule_evaluation/score_vector_bit.hpp"
#include "rule_evaluation_decomposable_common.hpp"

#include <algorithm>
#include <utility>

namespace seco {

    static inline float32 calculateLiftedQuality(float32 quality, uint32 numPredictions,
                                                 const ILiftFunction& liftFunction) {
        return (quality / numPredictions) * liftFunction.calculateLift(numPredictions);
    }

    /**
     * Allows to calculate the predictions of complete rules, as well as their overall quality, based on confusion
     * matrices, such that they optimize a heuristic that is applied to each output individually and takes into account
     * a specific lift function affecting the quality of rules, depending on how many labels they predict.
     *
     * @tparam StatisticVector The type of the vector that provides access to the confusion matrices
     */
    template<typename StatisticVector>
    class DecomposableCompleteRuleEvaluation final : public IRuleEvaluation<StatisticVector> {
        private:

            BitScoreVector<PartialIndexVector> scoreVector_;

            const std::unique_ptr<IHeuristic> heuristicPtr_;

            const std::unique_ptr<ILiftFunction> liftFunctionPtr_;

        public:

            /**
             * @param labelIndices      A reference to an object of type `PartialIndexVector` that provides access to
             *                          the indices of the labels for which the rules may predict
             * @param heuristicPtr      An unique pointer to an object of type `IHeuristic` that implements the
             *                          heuristic to be optimized
             * @param liftFunctionPtr   An unique pointer to an object of type `ILiftFunction` that should affect the
             *                          quality of rules, depending on how many labels they predict
             */
            DecomposableCompleteRuleEvaluation(const PartialIndexVector& labelIndices,
                                               std::unique_ptr<IHeuristic> heuristicPtr,
                                               std::unique_ptr<ILiftFunction> liftFunctionPtr)
                : scoreVector_(labelIndices, true), heuristicPtr_(std::move(heuristicPtr)),
                  liftFunctionPtr_(std::move(liftFunctionPtr)) {}

            const IScoreVector& calculateScores(View<uint32>::const_iterator majorityLabelIndicesBegin,
                                                View<uint32>::const_iterator majorityLabelIndicesEnd,
                                                const StatisticVector& confusionMatricesTotal,
                                                const StatisticVector& confusionMatricesCovered) override {
                uint32 numElements = scoreVector_.getNumElements();
                typename BitScoreVector<PartialIndexVector>::index_const_iterator indexIterator =
                  scoreVector_.indices_cbegin();
                typename StatisticVector::const_iterator totalIterator = confusionMatricesTotal.cbegin();
                typename StatisticVector::const_iterator coveredIterator = confusionMatricesCovered.cbegin();
                auto labelIterator =
                  createBinarySparseForwardIterator(majorityLabelIndicesBegin, majorityLabelIndicesEnd);
                float32 sumOfQualities = 0;
                uint32 previousIndex = 0;

                for (uint32 i = 0; i < numElements; i++) {
                    uint32 index = indexIterator[i];
                    std::advance(labelIterator, index - previousIndex);
                    scoreVector_.set(i, !(*labelIterator));
                    sumOfQualities +=
                      calculateOutputWiseQuality(totalIterator[index], coveredIterator[i], *heuristicPtr_);
                    previousIndex = index;
                }

                scoreVector_.quality = calculateLiftedQuality(sumOfQualities, numElements, *liftFunctionPtr_);
                return scoreVector_;
            }
    };

    /**
     * Allows to calculate the predictions of partial rules, as well as their overall quality, based on confusion
     * matrices, such that they optimize a heuristic that is applied to each output individually and takes into account
     * a specific lift function affecting the quality of rules, depending on how many labels they predict.
     *
     * @tparam StatisticVector  The type of the vector that provides access to the confusion matrices
     * @tparam IndexVector      The type of the vector that provides access to the indices of the labels for which
     *                          predictions should be calculated
     */
    template<typename StatisticVector, typename IndexVector>
    class DecomposablePartialRuleEvaluation final : public IRuleEvaluation<StatisticVector> {
        private:

            const IndexVector& labelIndices_;

            PartialIndexVector indexVector_;

            BitScoreVector<PartialIndexVector> scoreVector_;

            SparseArrayVector<std::pair<float32, bool>> sortedVector_;

            const std::unique_ptr<IHeuristic> heuristicPtr_;

            const std::unique_ptr<ILiftFunction> liftFunctionPtr_;

        public:

            /**
             * @param labelIndices      A reference to an object of template type `IndexVector` that provides access to
             *                          the indices of the labels for which the rules may predict
             * @param heuristicPtr      An unique pointer to an object of type `IHeuristic` that implements the
             *                          heuristic to be optimized
             * @param liftFunctionPtr   An unique pointer to an object of type `ILiftFunction` that should affect the
             *                          quality of rules, depending on how many labels they predict
             */
            DecomposablePartialRuleEvaluation(const IndexVector& labelIndices, std::unique_ptr<IHeuristic> heuristicPtr,
                                              std::unique_ptr<ILiftFunction> liftFunctionPtr)
                : labelIndices_(labelIndices), indexVector_(labelIndices.getNumElements()),
                  scoreVector_(indexVector_, false), sortedVector_(labelIndices.getNumElements()),
                  heuristicPtr_(std::move(heuristicPtr)), liftFunctionPtr_(std::move(liftFunctionPtr)) {}

            const IScoreVector& calculateScores(View<uint32>::const_iterator majorityLabelIndicesBegin,
                                                View<uint32>::const_iterator majorityLabelIndicesEnd,
                                                const StatisticVector& confusionMatricesTotal,
                                                const StatisticVector& confusionMatricesCovered) override {
                uint32 numElements = labelIndices_.getNumElements();
                typename IndexVector::const_iterator indexIterator = labelIndices_.cbegin();
                typename StatisticVector::const_iterator totalIterator = confusionMatricesTotal.cbegin();
                typename StatisticVector::const_iterator coveredIterator = confusionMatricesCovered.cbegin();
                auto labelIterator =
                  createBinarySparseForwardIterator(majorityLabelIndicesBegin, majorityLabelIndicesEnd);
                SparseArrayVector<std::pair<float32, bool>>::iterator sortedIterator = sortedVector_.begin();
                uint32 previousIndex = 0;

                for (uint32 i = 0; i < numElements; i++) {
                    uint32 index = indexIterator[i];
                    std::advance(labelIterator, index - previousIndex);
                    IndexedValue<std::pair<float32, bool>>& entry = sortedIterator[i];
                    std::pair<float32, bool>& pair = entry.value;
                    entry.index = index;
                    pair.first = calculateOutputWiseQuality(totalIterator[index], coveredIterator[i], *heuristicPtr_);
                    pair.second = !(*labelIterator);
                    previousIndex = index;
                }

                std::sort(sortedIterator, sortedVector_.end(),
                          [=](const IndexedValue<std::pair<float32, bool>>& a,
                              const IndexedValue<std::pair<float32, bool>>& b) {
                    return a.value.first > b.value.first;
                });

                float32 sumOfQualities = sortedIterator[0].value.first;
                float32 bestQuality = calculateLiftedQuality(sumOfQualities, 1, *liftFunctionPtr_);
                uint32 bestNumPredictions = 1;
                float32 maxLift = liftFunctionPtr_->getMaxLift(bestNumPredictions);

                for (uint32 i = 1; i < numElements; i++) {
                    uint32 numPredictions = i + 1;
                    sumOfQualities += sortedIterator[i].value.first;
                    float32 quality = calculateLiftedQuality(sumOfQualities, numPredictions, *liftFunctionPtr_);

                    if (quality > bestQuality) {
                        bestQuality = quality;
                        bestNumPredictions = numPredictions;

                        if (bestNumPredictions < numElements) {
                            maxLift = liftFunctionPtr_->getMaxLift(bestNumPredictions);
                        }
                    }

                    if (quality * maxLift < bestQuality) {
                        // Prunable by decomposability...
                        break;
                    }
                }

                indexVector_.setNumElements(bestNumPredictions, false);
                scoreVector_.quality = bestQuality;
                PartialIndexVector::iterator predictedIndexIterator = indexVector_.begin();

                for (uint32 i = 0; i < bestNumPredictions; i++) {
                    const IndexedValue<std::pair<float32, bool>>& entry = sortedIterator[i];
                    predictedIndexIterator[i] = entry.index;
                    scoreVector_.set(i, entry.value.second);
                }

                return scoreVector_;
            }
    };

    DecomposablePartialRuleEvaluationFactory::DecomposablePartialRuleEvaluationFactory(
      std::unique_ptr<IHeuristicFactory> heuristicFactoryPtr,
      std::unique_ptr<ILiftFunctionFactory> liftFunctionFactoryPtr)
        : heuristicFactoryPtr_(std::move(heuristicFactoryPtr)),
          liftFunctionFactoryPtr_(std::move(liftFunctionFactoryPtr)) {}

    std::unique_ptr<IRuleEvaluation<DenseConfusionMatrixVector<uint32>>>
      DecomposablePartialRuleEvaluationFactory::create(const DenseConfusionMatrixVector<uint32>& statisticVector,
                                                       const CompleteIndexVector& indexVector) const {
        std::unique_ptr<IHeuristic> heuristicPtr = heuristicFactoryPtr_->create();
        std::unique_ptr<ILiftFunction> liftFunctionPtr = liftFunctionFactoryPtr_->create();
        return std::make_unique<
          DecomposablePartialRuleEvaluation<DenseConfusionMatrixVector<uint32>, CompleteIndexVector>>(
          indexVector, std::move(heuristicPtr), std::move(liftFunctionPtr));
    }

    std::unique_ptr<IRuleEvaluation<DenseConfusionMatrixVector<uint32>>>
      DecomposablePartialRuleEvaluationFactory::create(const DenseConfusionMatrixVector<uint32>& statisticVector,
                                                       const PartialIndexVector& indexVector) const {
        std::unique_ptr<IHeuristic> heuristicPtr = heuristicFactoryPtr_->create();
        std::unique_ptr<ILiftFunction> liftFunctionPtr = liftFunctionFactoryPtr_->create();
        return std::make_unique<DecomposableCompleteRuleEvaluation<DenseConfusionMatrixVector<uint32>>>(
          indexVector, std::move(heuristicPtr), std::move(liftFunctionPtr));
    }

    std::unique_ptr<IRuleEvaluation<DenseConfusionMatrixVector<float32>>>
      DecomposablePartialRuleEvaluationFactory::create(const DenseConfusionMatrixVector<float32>& statisticVector,
                                                       const CompleteIndexVector& indexVector) const {
        std::unique_ptr<IHeuristic> heuristicPtr = heuristicFactoryPtr_->create();
        std::unique_ptr<ILiftFunction> liftFunctionPtr = liftFunctionFactoryPtr_->create();
        return std::make_unique<
          DecomposablePartialRuleEvaluation<DenseConfusionMatrixVector<float32>, CompleteIndexVector>>(
          indexVector, std::move(heuristicPtr), std::move(liftFunctionPtr));
    }

    std::unique_ptr<IRuleEvaluation<DenseConfusionMatrixVector<float32>>>
      DecomposablePartialRuleEvaluationFactory::create(const DenseConfusionMatrixVector<float32>& statisticVector,
                                                       const PartialIndexVector& indexVector) const {
        std::unique_ptr<IHeuristic> heuristicPtr = heuristicFactoryPtr_->create();
        std::unique_ptr<ILiftFunction> liftFunctionPtr = liftFunctionFactoryPtr_->create();
        return std::make_unique<DecomposableCompleteRuleEvaluation<DenseConfusionMatrixVector<float32>>>(
          indexVector, std::move(heuristicPtr), std::move(liftFunctionPtr));
    }

}
