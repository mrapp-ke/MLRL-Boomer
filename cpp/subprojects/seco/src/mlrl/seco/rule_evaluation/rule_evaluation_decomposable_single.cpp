#include "mlrl/seco/rule_evaluation/rule_evaluation_decomposable_single.hpp"

#include "mlrl/common/indices/index_vector_partial.hpp"
#include "mlrl/common/iterator/iterator_forward_sparse_binary.hpp"
#include "mlrl/common/rule_evaluation/score_vector_bit.hpp"
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

            BitScoreVector<PartialIndexVector> scoreVector_;

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
                                                const StatisticVector& statisticsUncovered,
                                                const StatisticVector& statisticsCovered) override {
                uint32 numElements = labelIndices_.getNumElements();
                auto indexIterator = labelIndices_.cbegin();
                auto tp = statisticsCovered.correct_counts_cbegin();
                auto fp = statisticsCovered.incorrect_counts_cbegin();
                auto fn = statisticsUncovered.correct_counts_cbegin();
                auto tn = statisticsUncovered.incorrect_counts_cbegin();
                uint32 bestIndex = indexIterator[0];
                float32 bestQuality = calculateOutputWiseQuality(tp[0], fp[0], fn[0], tn[0], *heuristicPtr_);

                for (uint32 i = 1; i < numElements; i++) {
                    uint32 index = indexIterator[i];
                    float32 quality = calculateOutputWiseQuality(tp[i], fp[i], fn[i], tn[i], *heuristicPtr_);

                    if (quality > bestQuality) {
                        bestIndex = index;
                        bestQuality = quality;
                    }
                }

                auto labelIterator =
                  createBinarySparseForwardIterator(majorityLabelIndicesBegin, majorityLabelIndicesEnd);
                std::advance(labelIterator, bestIndex);
                scoreVector_.set(0, !(*labelIterator));
                indexVector_.begin()[0] = bestIndex;
                scoreVector_.quality = bestQuality;
                return scoreVector_;
            }
    };

    DecomposableSingleOutputRuleEvaluationFactory::DecomposableSingleOutputRuleEvaluationFactory(
      std::unique_ptr<IHeuristicFactory> heuristicFactoryPtr)
        : heuristicFactoryPtr_(std::move(heuristicFactoryPtr)) {}

    std::unique_ptr<IRuleEvaluation<DenseDecomposableStatisticVectorView<uint32>>>
      DecomposableSingleOutputRuleEvaluationFactory::create(
        const DenseDecomposableStatisticVectorView<uint32>& statisticVector,
        const CompleteIndexVector& indexVector) const {
        std::unique_ptr<IHeuristic> heuristicPtr = heuristicFactoryPtr_->create();
        return std::make_unique<
          DecomposableSingleOutputRuleEvaluation<DenseDecomposableStatisticVectorView<uint32>, CompleteIndexVector>>(
          indexVector, std::move(heuristicPtr));
    }

    std::unique_ptr<IRuleEvaluation<DenseDecomposableStatisticVectorView<uint32>>>
      DecomposableSingleOutputRuleEvaluationFactory::create(
        const DenseDecomposableStatisticVectorView<uint32>& statisticVector,
        const PartialIndexVector& indexVector) const {
        std::unique_ptr<IHeuristic> heuristicPtr = heuristicFactoryPtr_->create();
        return std::make_unique<
          DecomposableSingleOutputRuleEvaluation<DenseDecomposableStatisticVectorView<uint32>, PartialIndexVector>>(
          indexVector, std::move(heuristicPtr));
    }

    std::unique_ptr<IRuleEvaluation<DenseDecomposableStatisticVectorView<float32>>>
      DecomposableSingleOutputRuleEvaluationFactory::create(
        const DenseDecomposableStatisticVectorView<float32>& statisticVector,
        const CompleteIndexVector& indexVector) const {
        std::unique_ptr<IHeuristic> heuristicPtr = heuristicFactoryPtr_->create();
        return std::make_unique<
          DecomposableSingleOutputRuleEvaluation<DenseDecomposableStatisticVectorView<float32>, CompleteIndexVector>>(
          indexVector, std::move(heuristicPtr));
    }

    std::unique_ptr<IRuleEvaluation<DenseDecomposableStatisticVectorView<float32>>>
      DecomposableSingleOutputRuleEvaluationFactory::create(
        const DenseDecomposableStatisticVectorView<float32>& statisticVector,
        const PartialIndexVector& indexVector) const {
        std::unique_ptr<IHeuristic> heuristicPtr = heuristicFactoryPtr_->create();
        return std::make_unique<
          DecomposableSingleOutputRuleEvaluation<DenseDecomposableStatisticVectorView<float32>, PartialIndexVector>>(
          indexVector, std::move(heuristicPtr));
    }

}
