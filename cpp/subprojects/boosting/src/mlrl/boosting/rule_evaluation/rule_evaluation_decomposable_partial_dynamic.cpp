#include "mlrl/boosting/rule_evaluation/rule_evaluation_decomposable_partial_dynamic.hpp"

#include "rule_evaluation_decomposable_complete_common.hpp"
#include "rule_evaluation_decomposable_partial_dynamic_common.hpp"

namespace boosting {

    /**
     * Allows to calculate the predictions of partial rules, which predict for a subset of the available outputs that is
     * determined dynamically, as well as their overall quality, based on the gradients and Hessians that are stored by
     * a vector using L1 and L2 regularization.
     *
     * @tparam StatisticVector  The type of the vector that provides access to the gradients and Hessians
     * @tparam IndexVector      The type of the vector that provides access to the indices of the outputs for which
     *                          predictions should be calculated
     */
    template<typename StatisticVector, typename IndexVector>
    class DecomposableDynamicPartialRuleEvaluation final : public IRuleEvaluation<StatisticVector> {
        private:

            const IndexVector& outputIndices_;

            PartialIndexVector indexVector_;

            DenseScoreVector<PartialIndexVector> scoreVector_;

            const float64 threshold_;

            const float64 exponent_;

            const float64 l1RegularizationWeight_;

            const float64 l2RegularizationWeight_;

        public:

            /**
             * @param outputIndices             A reference to an object of template type `IndexVector` that provides
             *                                  access to the indices of the outputs for which the rules may predict
             * @param threshold                 A threshold that affects for how many outputs the rule heads should
             *                                  predict
             * @param exponent                  An exponent that is used to weigh that estimated predictive quality for
             *                                  individual outputs
             * @param l1RegularizationWeight    The weight of the L1 regularization that is applied for calculating the
             *                                  scores to be predicted by rules
             * @param l2RegularizationWeight    The weight of the L2 regularization that is applied for calculating the
             *                                  scores to be predicted by rules
             */
            DecomposableDynamicPartialRuleEvaluation(const IndexVector& outputIndices, float32 threshold,
                                                     float32 exponent, float64 l1RegularizationWeight,
                                                     float64 l2RegularizationWeight)
                : outputIndices_(outputIndices), indexVector_(outputIndices.getNumElements()),
                  scoreVector_(indexVector_, true), threshold_(1.0 - threshold), exponent_(exponent),
                  l1RegularizationWeight_(l1RegularizationWeight), l2RegularizationWeight_(l2RegularizationWeight) {}

            const IScoreVector& calculateScores(StatisticVector& statisticVector) override {
                uint32 numElements = statisticVector.getNumElements();
                typename StatisticVector::const_iterator statisticIterator = statisticVector.cbegin();
                const std::pair<float64, float64> pair =
                  getMinAndMaxScore(statisticIterator, numElements, l1RegularizationWeight_, l2RegularizationWeight_);
                float64 minAbsScore = pair.first;
                float64 threshold = calculateThreshold(minAbsScore, pair.second, threshold_, exponent_);
                PartialIndexVector::iterator indexIterator = indexVector_.begin();
                DenseScoreVector<PartialIndexVector>::value_iterator valueIterator = scoreVector_.values_begin();
                typename IndexVector::const_iterator outputIndexIterator = outputIndices_.cbegin();
                float64 quality = 0;
                uint32 n = 0;

                for (uint32 i = 0; i < numElements; i++) {
                    const Tuple<float64>& tuple = statisticIterator[i];
                    float64 score = calculateOutputWiseScore(tuple.first, tuple.second, l1RegularizationWeight_,
                                                             l2RegularizationWeight_);

                    if (calculateWeightedScore(score, minAbsScore, exponent_) >= threshold) {
                        indexIterator[n] = outputIndexIterator[i];
                        valueIterator[n] = score;
                        quality += calculateOutputWiseQuality(score, tuple.first, tuple.second, l1RegularizationWeight_,
                                                              l2RegularizationWeight_);
                        n++;
                    }
                }

                indexVector_.setNumElements(n, false);
                scoreVector_.quality = quality;
                return scoreVector_;
            }
    };

    DecomposableDynamicPartialRuleEvaluationFactory::DecomposableDynamicPartialRuleEvaluationFactory(
      float32 threshold, float32 exponent, float64 l1RegularizationWeight, float64 l2RegularizationWeight)
        : threshold_(threshold), exponent_(exponent), l1RegularizationWeight_(l1RegularizationWeight),
          l2RegularizationWeight_(l2RegularizationWeight) {}

    std::unique_ptr<IRuleEvaluation<DenseDecomposableStatisticVector>>
      DecomposableDynamicPartialRuleEvaluationFactory::create(const DenseDecomposableStatisticVector& statisticVector,
                                                              const CompleteIndexVector& indexVector) const {
        return std::make_unique<
          DecomposableDynamicPartialRuleEvaluation<DenseDecomposableStatisticVector, CompleteIndexVector>>(
          indexVector, threshold_, exponent_, l1RegularizationWeight_, l2RegularizationWeight_);
    }

    std::unique_ptr<IRuleEvaluation<DenseDecomposableStatisticVector>>
      DecomposableDynamicPartialRuleEvaluationFactory::create(const DenseDecomposableStatisticVector& statisticVector,
                                                              const PartialIndexVector& indexVector) const {
        return std::make_unique<
          DecomposableCompleteRuleEvaluation<DenseDecomposableStatisticVector, PartialIndexVector>>(
          indexVector, l1RegularizationWeight_, l2RegularizationWeight_);
    }

    std::unique_ptr<IRuleEvaluation<SparseDecomposableStatisticVector>>
      DecomposableDynamicPartialRuleEvaluationFactory::create(const SparseDecomposableStatisticVector& statisticVector,
                                                              const CompleteIndexVector& indexVector) const {
        return std::make_unique<
          DecomposableDynamicPartialRuleEvaluation<SparseDecomposableStatisticVector, CompleteIndexVector>>(
          indexVector, threshold_, exponent_, l1RegularizationWeight_, l2RegularizationWeight_);
    }

    std::unique_ptr<IRuleEvaluation<SparseDecomposableStatisticVector>>
      DecomposableDynamicPartialRuleEvaluationFactory::create(const SparseDecomposableStatisticVector& statisticVector,
                                                              const PartialIndexVector& indexVector) const {
        return std::make_unique<
          DecomposableCompleteRuleEvaluation<SparseDecomposableStatisticVector, PartialIndexVector>>(
          indexVector, l1RegularizationWeight_, l2RegularizationWeight_);
    }

}
