#include "mlrl/seco/rule_evaluation/rule_evaluation_decomposable_single.hpp"

#include "mlrl/common/indices/index_vector_partial.hpp"
#include "mlrl/common/iterator/iterator_forward_sparse_binary.hpp"
#include "mlrl/common/rule_evaluation/score_vector_dense.hpp"
#include "rule_evaluation_decomposable_common.hpp"

namespace seco {

    /**
     * Allows to calculate the predictions of single-output rules, as well as their overall quality, based on confusion
     * matrices, such that they optimize a heuristic that is applied to each output individually.
     *
     * @tparam StatisticVector  The type of the vector that provides access to the confusion matrices
     * @tparam IndexVector      The type of the vector that provides access to the indices of the labels for which
     *                          predictions should be calculated
     */
    template<typename StatisticVector, typename IndexVector>
    class DecomposableSingleOutputRuleEvaluation final : public IRuleEvaluation<StatisticVector> {
        private:

            const IndexVector& labelIndices_;

            PartialIndexVector indexVector_;

            DenseScoreVector<float32, PartialIndexVector> scoreVector_;

            const std::unique_ptr<IHeuristic> heuristicPtr_;

        public:

            /**
             * @param labelIndices  A reference to an object of template type `IndexVector` that provides access to the
             *                      indices of the labels for which the rules may predict
             * @param heuristicPtr  An unique pointer to an object of type `IHeuristic` that implements the heuristic to
             *                      be optimized
             */
            DecomposableSingleOutputRuleEvaluation(const IndexVector& labelIndices,
                                                   std::unique_ptr<IHeuristic> heuristicPtr)
                : labelIndices_(labelIndices), indexVector_(1), scoreVector_(indexVector_, true),
                  heuristicPtr_(std::move(heuristicPtr)) {}

            const IScoreVector& calculateScores(View<uint32>::const_iterator majorityLabelIndicesBegin,
                                                View<uint32>::const_iterator majorityLabelIndicesEnd,
                                                const StatisticVector& confusionMatricesTotal,
                                                const StatisticVector& confusionMatricesCovered) override {
                uint32 numElements = labelIndices_.getNumElements();
                typename IndexVector::const_iterator indexIterator = labelIndices_.cbegin();
                typename StatisticVector::const_iterator totalIterator = confusionMatricesTotal.cbegin();
                typename StatisticVector::const_iterator coveredIterator = confusionMatricesCovered.cbegin();
                uint32 bestIndex = indexIterator[0];
                float32 bestQuality =
                  calculateOutputWiseQuality(totalIterator[bestIndex], coveredIterator[0], *heuristicPtr_);

                for (uint32 i = 1; i < numElements; i++) {
                    uint32 index = indexIterator[i];
                    float32 quality =
                      calculateOutputWiseQuality(totalIterator[index], coveredIterator[i], *heuristicPtr_);

                    if (quality > bestQuality) {
                        bestIndex = index;
                        bestQuality = quality;
                    }
                }

                auto labelIterator =
                  createBinarySparseForwardIterator(majorityLabelIndicesBegin, majorityLabelIndicesEnd);
                std::advance(labelIterator, bestIndex);
                scoreVector_.values_begin()[0] = (float32) !(*labelIterator);
                indexVector_.begin()[0] = bestIndex;
                scoreVector_.quality = bestQuality;
                return scoreVector_;
            }
    };

    DecomposableSingleOutputRuleEvaluationFactory::DecomposableSingleOutputRuleEvaluationFactory(
      std::unique_ptr<IHeuristicFactory> heuristicFactoryPtr)
        : heuristicFactoryPtr_(std::move(heuristicFactoryPtr)) {}

    std::unique_ptr<IRuleEvaluation<DenseConfusionMatrixVector<uint32>>>
      DecomposableSingleOutputRuleEvaluationFactory::create(const DenseConfusionMatrixVector<uint32>& statisticVector,
                                                            const CompleteIndexVector& indexVector) const {
        std::unique_ptr<IHeuristic> heuristicPtr = heuristicFactoryPtr_->create();
        return std::make_unique<
          DecomposableSingleOutputRuleEvaluation<DenseConfusionMatrixVector<uint32>, CompleteIndexVector>>(
          indexVector, std::move(heuristicPtr));
    }

    std::unique_ptr<IRuleEvaluation<DenseConfusionMatrixVector<uint32>>>
      DecomposableSingleOutputRuleEvaluationFactory::create(const DenseConfusionMatrixVector<uint32>& statisticVector,
                                                            const PartialIndexVector& indexVector) const {
        std::unique_ptr<IHeuristic> heuristicPtr = heuristicFactoryPtr_->create();
        return std::make_unique<
          DecomposableSingleOutputRuleEvaluation<DenseConfusionMatrixVector<uint32>, PartialIndexVector>>(
          indexVector, std::move(heuristicPtr));
    }

    std::unique_ptr<IRuleEvaluation<DenseConfusionMatrixVector<float32>>>
      DecomposableSingleOutputRuleEvaluationFactory::create(const DenseConfusionMatrixVector<float32>& statisticVector,
                                                            const CompleteIndexVector& indexVector) const {
        std::unique_ptr<IHeuristic> heuristicPtr = heuristicFactoryPtr_->create();
        return std::make_unique<
          DecomposableSingleOutputRuleEvaluation<DenseConfusionMatrixVector<float32>, CompleteIndexVector>>(
          indexVector, std::move(heuristicPtr));
    }

    std::unique_ptr<IRuleEvaluation<DenseConfusionMatrixVector<float32>>>
      DecomposableSingleOutputRuleEvaluationFactory::create(const DenseConfusionMatrixVector<float32>& statisticVector,
                                                            const PartialIndexVector& indexVector) const {
        std::unique_ptr<IHeuristic> heuristicPtr = heuristicFactoryPtr_->create();
        return std::make_unique<
          DecomposableSingleOutputRuleEvaluation<DenseConfusionMatrixVector<float32>, PartialIndexVector>>(
          indexVector, std::move(heuristicPtr));
    }

}
