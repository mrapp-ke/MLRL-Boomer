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

            typedef typename StatisticVector::statistic_type statistic_type;

            const IndexVector& outputIndices_;

            PartialIndexVector indexVector_;

            DenseScoreVector<statistic_type, PartialIndexVector> scoreVector_;

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
                typename StatisticVector::gradient_const_iterator gradientIterator = statisticVector.gradients_cbegin();
                typename StatisticVector::hessian_const_iterator hessianIterator = statisticVector.hessians_cbegin();
                statistic_type bestScore = calculateOutputWiseScore(gradientIterator[0], hessianIterator[0],
                                                                    l1RegularizationWeight_, l2RegularizationWeight_);
                uint32 bestIndex = 0;

                for (uint32 i = 1; i < numElements; i++) {
                    statistic_type score = calculateOutputWiseScore(gradientIterator[i], hessianIterator[i],
                                                                    l1RegularizationWeight_, l2RegularizationWeight_);

                    if (std::abs(score) > std::abs(bestScore)) {
                        bestIndex = i;
                        bestScore = score;
                    }
                }

                typename DenseScoreVector<statistic_type, PartialIndexVector>::value_iterator valueIterator =
                  scoreVector_.values_begin();
                valueIterator[0] = bestScore;
                indexVector_.begin()[0] = outputIndices_.cbegin()[bestIndex];
                scoreVector_.quality =
                  calculateOutputWiseQuality(bestScore, gradientIterator[bestIndex], hessianIterator[bestIndex],
                                             l1RegularizationWeight_, l2RegularizationWeight_);
                return scoreVector_;
            }
    };

    DecomposableSingleOutputRuleEvaluationFactory::DecomposableSingleOutputRuleEvaluationFactory(
      float32 l1RegularizationWeight, float32 l2RegularizationWeight)
        : l1RegularizationWeight_(l1RegularizationWeight), l2RegularizationWeight_(l2RegularizationWeight) {}

    std::unique_ptr<IRuleEvaluation<DenseDecomposableStatisticVectorView<float32>>>
      DecomposableSingleOutputRuleEvaluationFactory::create(
        const DenseDecomposableStatisticVectorView<float32>& statisticVector,
        const CompleteIndexVector& indexVector) const {
        return std::make_unique<
          DecomposableSingleOutputRuleEvaluation<DenseDecomposableStatisticVectorView<float32>, CompleteIndexVector>>(
          indexVector, l1RegularizationWeight_, l2RegularizationWeight_);
    }

    std::unique_ptr<IRuleEvaluation<DenseDecomposableStatisticVectorView<float32>>>
      DecomposableSingleOutputRuleEvaluationFactory::create(
        const DenseDecomposableStatisticVectorView<float32>& statisticVector,
        const PartialIndexVector& indexVector) const {
        return std::make_unique<
          DecomposableSingleOutputRuleEvaluation<DenseDecomposableStatisticVectorView<float32>, PartialIndexVector>>(
          indexVector, l1RegularizationWeight_, l2RegularizationWeight_);
    }

    std::unique_ptr<IRuleEvaluation<DenseDecomposableStatisticVectorView<float64>>>
      DecomposableSingleOutputRuleEvaluationFactory::create(
        const DenseDecomposableStatisticVectorView<float64>& statisticVector,
        const CompleteIndexVector& indexVector) const {
        return std::make_unique<
          DecomposableSingleOutputRuleEvaluation<DenseDecomposableStatisticVectorView<float64>, CompleteIndexVector>>(
          indexVector, l1RegularizationWeight_, l2RegularizationWeight_);
    }

    std::unique_ptr<IRuleEvaluation<DenseDecomposableStatisticVectorView<float64>>>
      DecomposableSingleOutputRuleEvaluationFactory::create(
        const DenseDecomposableStatisticVectorView<float64>& statisticVector,
        const PartialIndexVector& indexVector) const {
        return std::make_unique<
          DecomposableSingleOutputRuleEvaluation<DenseDecomposableStatisticVectorView<float64>, PartialIndexVector>>(
          indexVector, l1RegularizationWeight_, l2RegularizationWeight_);
    }

    std::unique_ptr<IRuleEvaluation<SparseDecomposableStatisticVectorView<float32, uint32>>>
      DecomposableSingleOutputRuleEvaluationFactory::create(
        const SparseDecomposableStatisticVectorView<float32, uint32>& statisticVector,
        const CompleteIndexVector& indexVector) const {
        return std::make_unique<DecomposableSingleOutputRuleEvaluation<
          SparseDecomposableStatisticVectorView<float32, uint32>, CompleteIndexVector>>(
          indexVector, l1RegularizationWeight_, l2RegularizationWeight_);
    }

    std::unique_ptr<IRuleEvaluation<SparseDecomposableStatisticVectorView<float32, uint32>>>
      DecomposableSingleOutputRuleEvaluationFactory::create(
        const SparseDecomposableStatisticVectorView<float32, uint32>& statisticVector,
        const PartialIndexVector& indexVector) const {
        return std::make_unique<DecomposableSingleOutputRuleEvaluation<
          SparseDecomposableStatisticVectorView<float32, uint32>, PartialIndexVector>>(
          indexVector, l1RegularizationWeight_, l2RegularizationWeight_);
    }

    std::unique_ptr<IRuleEvaluation<SparseDecomposableStatisticVectorView<float32, float32>>>
      DecomposableSingleOutputRuleEvaluationFactory::create(
        const SparseDecomposableStatisticVectorView<float32, float32>& statisticVector,
        const CompleteIndexVector& indexVector) const {
        return std::make_unique<DecomposableSingleOutputRuleEvaluation<
          SparseDecomposableStatisticVectorView<float32, float32>, CompleteIndexVector>>(
          indexVector, l1RegularizationWeight_, l2RegularizationWeight_);
    }

    std::unique_ptr<IRuleEvaluation<SparseDecomposableStatisticVectorView<float32, float32>>>
      DecomposableSingleOutputRuleEvaluationFactory::create(
        const SparseDecomposableStatisticVectorView<float32, float32>& statisticVector,
        const PartialIndexVector& indexVector) const {
        return std::make_unique<DecomposableSingleOutputRuleEvaluation<
          SparseDecomposableStatisticVectorView<float32, float32>, PartialIndexVector>>(
          indexVector, l1RegularizationWeight_, l2RegularizationWeight_);
    }

    std::unique_ptr<IRuleEvaluation<SparseDecomposableStatisticVectorView<float64, uint32>>>
      DecomposableSingleOutputRuleEvaluationFactory::create(
        const SparseDecomposableStatisticVectorView<float64, uint32>& statisticVector,
        const CompleteIndexVector& indexVector) const {
        return std::make_unique<DecomposableSingleOutputRuleEvaluation<
          SparseDecomposableStatisticVectorView<float64, uint32>, CompleteIndexVector>>(
          indexVector, l1RegularizationWeight_, l2RegularizationWeight_);
    }

    std::unique_ptr<IRuleEvaluation<SparseDecomposableStatisticVectorView<float64, uint32>>>
      DecomposableSingleOutputRuleEvaluationFactory::create(
        const SparseDecomposableStatisticVectorView<float64, uint32>& statisticVector,
        const PartialIndexVector& indexVector) const {
        return std::make_unique<DecomposableSingleOutputRuleEvaluation<
          SparseDecomposableStatisticVectorView<float64, uint32>, PartialIndexVector>>(
          indexVector, l1RegularizationWeight_, l2RegularizationWeight_);
    }

    std::unique_ptr<IRuleEvaluation<SparseDecomposableStatisticVectorView<float64, float32>>>
      DecomposableSingleOutputRuleEvaluationFactory::create(
        const SparseDecomposableStatisticVectorView<float64, float32>& statisticVector,
        const CompleteIndexVector& indexVector) const {
        return std::make_unique<DecomposableSingleOutputRuleEvaluation<
          SparseDecomposableStatisticVectorView<float64, float32>, CompleteIndexVector>>(
          indexVector, l1RegularizationWeight_, l2RegularizationWeight_);
    }

    std::unique_ptr<IRuleEvaluation<SparseDecomposableStatisticVectorView<float64, float32>>>
      DecomposableSingleOutputRuleEvaluationFactory::create(
        const SparseDecomposableStatisticVectorView<float64, float32>& statisticVector,
        const PartialIndexVector& indexVector) const {
        return std::make_unique<DecomposableSingleOutputRuleEvaluation<
          SparseDecomposableStatisticVectorView<float64, float32>, PartialIndexVector>>(
          indexVector, l1RegularizationWeight_, l2RegularizationWeight_);
    }

}
