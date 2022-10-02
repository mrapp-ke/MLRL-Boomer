#include "seco/rule_evaluation/rule_evaluation_label_wise_partial.hpp"
#include "common/data/tuple.hpp"
#include "common/data/vector_sparse_array.hpp"
#include "common/indices/index_vector_partial.hpp"
#include "common/iterator/binary_forward_iterator.hpp"
#include "common/rule_evaluation/score_vector_dense.hpp"
#include "rule_evaluation_label_wise_common.hpp"
#include <algorithm>


namespace seco {

    static inline float64 calculateLiftedQuality(float64 quality, uint32 numPredictions,
                                                 const ILiftFunction& liftFunction) {
        return (quality / numPredictions) * liftFunction.calculateLift(numPredictions);
    }

    /**
     * Allows to calculate the predictions of complete rules, as well as their overall quality, such that they optimize
     * a heuristic that is applied using label-wise averaging and taking a specific lift function, which affects the
     * quality of rules, depending on how many labels they predict, into account.
     */
    class LabelWiseCompleteRuleEvaluation final : public IRuleEvaluation {

        private:

            DenseScoreVector<PartialIndexVector> scoreVector_;

            std::unique_ptr<IHeuristic> heuristicPtr_;

            std::unique_ptr<ILiftFunction> liftFunctionPtr_;

        public:

            /**
             * @param labelIndices      A reference to an object of type `PartialIndexVector` that provides access to
             *                          the indices of the labels for which the rules may predict
             * @param heuristicPtr      An unique pointer to an object of type `IHeuristic` that implements the
             *                          heuristic to be optimized
             * @param liftFunctionPtr   An unique pointer to an object of type `ILiftFunction` that should affect the
             *                          quality of rules, depending on how many labels they predict
             */
            LabelWiseCompleteRuleEvaluation(const PartialIndexVector& labelIndices,
                                            std::unique_ptr<IHeuristic> heuristicPtr,
                                            std::unique_ptr<ILiftFunction> liftFunctionPtr)
                : scoreVector_(DenseScoreVector<PartialIndexVector>(labelIndices, true)),
                  heuristicPtr_(std::move(heuristicPtr)), liftFunctionPtr_(std::move(liftFunctionPtr)) {

            }

            const IScoreVector& calculateScores(const VectorConstView<uint32>& majorityLabelIndices,
                                                const DenseConfusionMatrixVector& confusionMatricesTotal,
                                                const DenseConfusionMatrixVector& confusionMatricesCovered) override {
                uint32 numElements = scoreVector_.getNumElements();
                DenseScoreVector<PartialIndexVector>::index_const_iterator indexIterator =
                    scoreVector_.indices_cbegin();
                DenseConfusionMatrixVector::const_iterator totalIterator = confusionMatricesTotal.cbegin();
                DenseConfusionMatrixVector::const_iterator coveredIterator = confusionMatricesCovered.cbegin();
                auto labelIterator = make_binary_forward_iterator(majorityLabelIndices.cbegin(),
                                                                  majorityLabelIndices.cend());
                DenseScoreVector<PartialIndexVector>::score_iterator scoreIterator = scoreVector_.scores_begin();
                float64 sumOfQualities = 0;
                uint32 previousIndex = 0;

                for (uint32 i = 0; i < numElements; i++) {
                    uint32 index = indexIterator[i];
                    std::advance(labelIterator, index - previousIndex);
                    scoreIterator[i] = (float64) !(*labelIterator);
                    sumOfQualities += (1 - calculateLabelWiseQuality(totalIterator[index], coveredIterator[i],
                                                                     *heuristicPtr_));
                    previousIndex = index;
                }

                scoreVector_.overallQualityScore = (1 - calculateLiftedQuality(sumOfQualities, numElements,
                                                                               *liftFunctionPtr_));
                return scoreVector_;
            }

    };

    /**
     * Allows to calculate the predictions of partial rules, as well as their overall quality, such that they optimize a
     * heuristic that is applied using label-wise averaging and taking a specific lift function, which affects the
     * quality of rules, depending on how many labels they predict, into account.
     *
     * @tparam T The type of the vector that provides access to the labels for which predictions should be calculated
     */
    template<typename T>
    class LabelWisePartialRuleEvaluation final : public IRuleEvaluation {

        private:

            const T& labelIndices_;

            PartialIndexVector indexVector_;

            DenseScoreVector<PartialIndexVector> scoreVector_;

            SparseArrayVector<Tuple<float64>> sortedVector_;

            std::unique_ptr<IHeuristic> heuristicPtr_;

            std::unique_ptr<ILiftFunction> liftFunctionPtr_;

        public:

            /**
             * @param labelIndices      A reference to an object of template type `T` that provides access to the
             *                          indices of the labels for which the rules may predict
             * @param heuristicPtr      An unique pointer to an object of type `IHeuristic` that implements the
             *                          heuristic to be optimized
             * @param liftFunctionPtr   An unique pointer to an object of type `ILiftFunction` that should affect the
             *                          quality of rules, depending on how many labels they predict
             */
            LabelWisePartialRuleEvaluation(const T& labelIndices, std::unique_ptr<IHeuristic> heuristicPtr,
                                           std::unique_ptr<ILiftFunction> liftFunctionPtr)
                : labelIndices_(labelIndices), indexVector_(PartialIndexVector(labelIndices.getNumElements())),
                  scoreVector_(DenseScoreVector<PartialIndexVector>(indexVector_, false)),
                  sortedVector_(SparseArrayVector<Tuple<float64>>(labelIndices.getNumElements())),
                  heuristicPtr_(std::move(heuristicPtr)), liftFunctionPtr_(std::move(liftFunctionPtr)) {

            }

            const IScoreVector& calculateScores(const VectorConstView<uint32>& majorityLabelIndices,
                                                const DenseConfusionMatrixVector& confusionMatricesTotal,
                                                const DenseConfusionMatrixVector& confusionMatricesCovered) override {
                uint32 numElements = labelIndices_.getNumElements();
                typename T::const_iterator indexIterator = labelIndices_.cbegin();
                DenseConfusionMatrixVector::const_iterator totalIterator = confusionMatricesTotal.cbegin();
                DenseConfusionMatrixVector::const_iterator coveredIterator = confusionMatricesCovered.cbegin();
                auto labelIterator = make_binary_forward_iterator(majorityLabelIndices.cbegin(),
                                                                  majorityLabelIndices.cend());
                SparseArrayVector<Tuple<float64>>::iterator sortedIterator = sortedVector_.begin();
                uint32 previousIndex = 0;

                for (uint32 i = 0; i < numElements; i++) {
                    uint32 index = indexIterator[i];
                    std::advance(labelIterator, index - previousIndex);
                    IndexedValue<Tuple<float64>>& entry = sortedIterator[i];
                    Tuple<float64>& tuple = entry.value;
                    entry.index = index;
                    tuple.first = calculateLabelWiseQuality(totalIterator[index], coveredIterator[i], *heuristicPtr_);
                    tuple.second = (float64) !(*labelIterator);
                    previousIndex = index;
                }

                std::sort(sortedIterator, sortedVector_.end(), [=](const IndexedValue<Tuple<float64>>& a,
                                                                   const IndexedValue<Tuple<float64>>& b) {
                    return a.value.first < b.value.first;
                });

                float64 sumOfQualities = (1 - sortedIterator[0].value.first);
                float64 bestQuality = calculateLiftedQuality(sumOfQualities, 1, *liftFunctionPtr_);
                uint32 bestNumPredictions = 1;
                float64 maxLift = liftFunctionPtr_->getMaxLift(bestNumPredictions);

                for (uint32 i = 1; i < numElements; i++) {
                    uint32 numPredictions = i + 1;
                    sumOfQualities += (1 - sortedIterator[i].value.first);
                    float64 quality = calculateLiftedQuality(sumOfQualities, numPredictions, *liftFunctionPtr_);

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
                scoreVector_.overallQualityScore = (1 - bestQuality);
                DenseScoreVector<PartialIndexVector>::score_iterator scoreIterator = scoreVector_.scores_begin();
                PartialIndexVector::iterator predictedIndexIterator = indexVector_.begin();

                for (uint32 i = 0; i < bestNumPredictions; i++) {
                    const IndexedValue<Tuple<float64>>& entry = sortedIterator[i];
                    predictedIndexIterator[i] = entry.index;
                    scoreIterator[i] = entry.value.second;
                }

                return scoreVector_;
            }

    };

    LabelWisePartialRuleEvaluationFactory::LabelWisePartialRuleEvaluationFactory(
            std::unique_ptr<IHeuristicFactory> heuristicFactoryPtr,
            std::unique_ptr<ILiftFunctionFactory> liftFunctionFactoryPtr)
        : heuristicFactoryPtr_(std::move(heuristicFactoryPtr)),
          liftFunctionFactoryPtr_(std::move(liftFunctionFactoryPtr)) {

    }

    std::unique_ptr<IRuleEvaluation> LabelWisePartialRuleEvaluationFactory::create(
            const CompleteIndexVector& indexVector) const {
        std::unique_ptr<IHeuristic> heuristicPtr = heuristicFactoryPtr_->create();
        std::unique_ptr<ILiftFunction> liftFunctionPtr = liftFunctionFactoryPtr_->create();
        return std::make_unique<LabelWisePartialRuleEvaluation<CompleteIndexVector>>(indexVector,
                                                                                     std::move(heuristicPtr),
                                                                                     std::move(liftFunctionPtr));
    }

    std::unique_ptr<IRuleEvaluation> LabelWisePartialRuleEvaluationFactory::create(
            const PartialIndexVector& indexVector) const {
        std::unique_ptr<IHeuristic> heuristicPtr = heuristicFactoryPtr_->create();
        std::unique_ptr<ILiftFunction> liftFunctionPtr = liftFunctionFactoryPtr_->create();
        return std::make_unique<LabelWiseCompleteRuleEvaluation>(indexVector, std::move(heuristicPtr),
                                                                 std::move(liftFunctionPtr));
    }

}
