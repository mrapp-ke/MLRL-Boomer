#include "mlrl/boosting/rule_evaluation/rule_evaluation_decomposable_single.hpp"

#include "mlrl/common/rule_evaluation/score_vector_dense.hpp"
#include "rule_evaluation_decomposable_common.hpp"

namespace boosting {

    /**
     * Allows to calculate the predictions of single-output rules, as well as their overall quality, based on the
     * gradients and Hessians that are stored by a vector using L1 and L2 regularization.
     *
     * @tparam StatisticVector  The type of the vector that provides access to the gradients and Hessians
     * @tparam IndexVector      The type of the vector that provides access to the indices of the outputs for which
     *                          predictions should be calculated
     */
    template<typename StatisticVector, typename IndexVector>
    class DecomposableSingleOutputRuleEvaluation final : public IRuleEvaluation<StatisticVector> {
        private:

            const IndexVector& outputIndices_;

            PartialIndexVector indexVector_;

            DenseScoreVector<PartialIndexVector> scoreVector_;

            const float32 l1RegularizationWeight_;

            const float32 l2RegularizationWeight_;

        public:

            /**
             * @param outputIndices             A reference to an object of template type `IndexVector` that provides
             *                                  access to the indices of the outputs for which the rules may predict
             * @param l1RegularizationWeight    The weight of the L1 regularization that is applied for calculating the
             *                                  scores to be predicted by rules
             * @param l2RegularizationWeight    The weight of the L2 regularization that is applied for calculating the
             *                                  scores to be predicted by rules
             */
            DecomposableSingleOutputRuleEvaluation(const IndexVector& outputIndices, float32 l1RegularizationWeight,
                                                   float32 l2RegularizationWeight)
                : outputIndices_(outputIndices), indexVector_(1), scoreVector_(indexVector_, true),
                  l1RegularizationWeight_(l1RegularizationWeight), l2RegularizationWeight_(l2RegularizationWeight) {}

            const IScoreVector& calculateScores(StatisticVector& statisticVector) override {
                uint32 numElements = statisticVector.getNumElements();
                typename StatisticVector::const_iterator statisticIterator = statisticVector.cbegin();
                const typename StatisticVector::value_type& firstStatistic = statisticIterator[0];
                typename StatisticVector::value_type::statistic_type bestScore = calculateOutputWiseScore(
                  firstStatistic.gradient, firstStatistic.hessian, l1RegularizationWeight_, l2RegularizationWeight_);
                uint32 bestIndex = 0;

                for (uint32 i = 1; i < numElements; i++) {
                    const typename StatisticVector::value_type& statistic = statisticIterator[i];
                    typename StatisticVector::value_type::statistic_type score = calculateOutputWiseScore(
                      statistic.gradient, statistic.hessian, l1RegularizationWeight_, l2RegularizationWeight_);

                    if (std::abs(score) > std::abs(bestScore)) {
                        bestIndex = i;
                        bestScore = score;
                    }
                }

                DenseScoreVector<PartialIndexVector>::value_iterator valueIterator = scoreVector_.values_begin();
                valueIterator[0] = bestScore;
                indexVector_.begin()[0] = outputIndices_.cbegin()[bestIndex];
                const typename StatisticVector::value_type& bestStatistic = statisticIterator[bestIndex];
                scoreVector_.quality =
                  calculateOutputWiseQuality(bestScore, bestStatistic.gradient, bestStatistic.hessian,
                                             l1RegularizationWeight_, l2RegularizationWeight_);
                return scoreVector_;
            }
    };

    DecomposableSingleOutputRuleEvaluationFactory::DecomposableSingleOutputRuleEvaluationFactory(
      float32 l1RegularizationWeight, float32 l2RegularizationWeight)
        : l1RegularizationWeight_(l1RegularizationWeight), l2RegularizationWeight_(l2RegularizationWeight) {}

    std::unique_ptr<IRuleEvaluation<DenseDecomposableStatisticVector<float64>>>
      DecomposableSingleOutputRuleEvaluationFactory::create(
        const DenseDecomposableStatisticVector<float64>& statisticVector,
        const CompleteIndexVector& indexVector) const {
        return std::make_unique<
          DecomposableSingleOutputRuleEvaluation<DenseDecomposableStatisticVector<float64>, CompleteIndexVector>>(
          indexVector, l1RegularizationWeight_, l2RegularizationWeight_);
    }

    std::unique_ptr<IRuleEvaluation<DenseDecomposableStatisticVector<float64>>>
      DecomposableSingleOutputRuleEvaluationFactory::create(
        const DenseDecomposableStatisticVector<float64>& statisticVector, const PartialIndexVector& indexVector) const {
        return std::make_unique<
          DecomposableSingleOutputRuleEvaluation<DenseDecomposableStatisticVector<float64>, PartialIndexVector>>(
          indexVector, l1RegularizationWeight_, l2RegularizationWeight_);
    }

    std::unique_ptr<IRuleEvaluation<SparseDecomposableStatisticVector<float64, uint32>>>
      DecomposableSingleOutputRuleEvaluationFactory::create(
        const SparseDecomposableStatisticVector<float64, uint32>& statisticVector,
        const CompleteIndexVector& indexVector) const {
        return std::make_unique<DecomposableSingleOutputRuleEvaluation<
          SparseDecomposableStatisticVector<float64, uint32>, CompleteIndexVector>>(
          indexVector, l1RegularizationWeight_, l2RegularizationWeight_);
    }

    std::unique_ptr<IRuleEvaluation<SparseDecomposableStatisticVector<float64, uint32>>>
      DecomposableSingleOutputRuleEvaluationFactory::create(
        const SparseDecomposableStatisticVector<float64, uint32>& statisticVector,
        const PartialIndexVector& indexVector) const {
        return std::make_unique<DecomposableSingleOutputRuleEvaluation<
          SparseDecomposableStatisticVector<float64, uint32>, PartialIndexVector>>(indexVector, l1RegularizationWeight_,
                                                                                   l2RegularizationWeight_);
    }

    std::unique_ptr<IRuleEvaluation<SparseDecomposableStatisticVector<float64, float32>>>
      DecomposableSingleOutputRuleEvaluationFactory::create(
        const SparseDecomposableStatisticVector<float64, float32>& statisticVector,
        const CompleteIndexVector& indexVector) const {
        return std::make_unique<DecomposableSingleOutputRuleEvaluation<
          SparseDecomposableStatisticVector<float64, float32>, CompleteIndexVector>>(
          indexVector, l1RegularizationWeight_, l2RegularizationWeight_);
    }

    std::unique_ptr<IRuleEvaluation<SparseDecomposableStatisticVector<float64, float32>>>
      DecomposableSingleOutputRuleEvaluationFactory::create(
        const SparseDecomposableStatisticVector<float64, float32>& statisticVector,
        const PartialIndexVector& indexVector) const {
        return std::make_unique<DecomposableSingleOutputRuleEvaluation<
          SparseDecomposableStatisticVector<float64, float32>, PartialIndexVector>>(
          indexVector, l1RegularizationWeight_, l2RegularizationWeight_);
    }

}
