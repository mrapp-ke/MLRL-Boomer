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

            typedef typename StatisticVector::statistic_type statistic_type;

            const IndexVector& outputIndices_;

            PartialIndexVector indexVector_;

            DenseScoreVector<statistic_type, PartialIndexVector> scoreVector_;

            const float32 threshold_;

            const float32 exponent_;

            const float32 l1RegularizationWeight_;

            const float32 l2RegularizationWeight_;

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
                                                     float32 exponent, float32 l1RegularizationWeight,
                                                     float32 l2RegularizationWeight)
                : outputIndices_(outputIndices), indexVector_(outputIndices.getNumElements()),
                  scoreVector_(indexVector_, true), threshold_(1.0f - threshold), exponent_(exponent),
                  l1RegularizationWeight_(l1RegularizationWeight), l2RegularizationWeight_(l2RegularizationWeight) {}

            const IScoreVector& calculateScores(StatisticVector& statisticVector) override {
                uint32 numElements = statisticVector.getNumElements();
                typename StatisticVector::const_iterator statisticIterator = statisticVector.cbegin();
                const std::pair<statistic_type, statistic_type> pair =
                  getMinAndMaxScore(statisticIterator, numElements, l1RegularizationWeight_, l2RegularizationWeight_);
                statistic_type minAbsScore = pair.first;
                statistic_type threshold = calculateThreshold(minAbsScore, pair.second, threshold_, exponent_);
                PartialIndexVector::iterator indexIterator = indexVector_.begin();
                typename DenseScoreVector<statistic_type, PartialIndexVector>::value_iterator valueIterator =
                  scoreVector_.values_begin();
                typename IndexVector::const_iterator outputIndexIterator = outputIndices_.cbegin();
                statistic_type quality = 0;
                uint32 n = 0;

                for (uint32 i = 0; i < numElements; i++) {
                    const Statistic<statistic_type>& statistic = statisticIterator[i];
                    statistic_type score = calculateOutputWiseScore(statistic.gradient, statistic.hessian,
                                                                    l1RegularizationWeight_, l2RegularizationWeight_);

                    if (calculateWeightedScore(score, minAbsScore, exponent_) >= threshold) {
                        indexIterator[n] = outputIndexIterator[i];
                        valueIterator[n] = score;
                        quality += calculateOutputWiseQuality(score, statistic.gradient, statistic.hessian,
                                                              l1RegularizationWeight_, l2RegularizationWeight_);
                        n++;
                    }
                }

                indexVector_.setNumElements(n, false);
                scoreVector_.quality = quality;
                return scoreVector_;
            }
    };

    DecomposableDynamicPartialRuleEvaluationFactory::DecomposableDynamicPartialRuleEvaluationFactory(
      float32 threshold, float32 exponent, float32 l1RegularizationWeight, float32 l2RegularizationWeight)
        : threshold_(threshold), exponent_(exponent), l1RegularizationWeight_(l1RegularizationWeight),
          l2RegularizationWeight_(l2RegularizationWeight) {}

    std::unique_ptr<IRuleEvaluation<DenseDecomposableStatisticVector<float32>>>
      DecomposableDynamicPartialRuleEvaluationFactory::create(
        const DenseDecomposableStatisticVector<float32>& statisticVector,
        const CompleteIndexVector& indexVector) const {
        return std::make_unique<
          DecomposableDynamicPartialRuleEvaluation<DenseDecomposableStatisticVector<float32>, CompleteIndexVector>>(
          indexVector, threshold_, exponent_, l1RegularizationWeight_, l2RegularizationWeight_);
    }

    std::unique_ptr<IRuleEvaluation<DenseDecomposableStatisticVector<float32>>>
      DecomposableDynamicPartialRuleEvaluationFactory::create(
        const DenseDecomposableStatisticVector<float32>& statisticVector, const PartialIndexVector& indexVector) const {
        return std::make_unique<
          DecomposableCompleteRuleEvaluation<DenseDecomposableStatisticVector<float32>, PartialIndexVector>>(
          indexVector, l1RegularizationWeight_, l2RegularizationWeight_);
    }

    std::unique_ptr<IRuleEvaluation<DenseDecomposableStatisticVector<float64>>>
      DecomposableDynamicPartialRuleEvaluationFactory::create(
        const DenseDecomposableStatisticVector<float64>& statisticVector,
        const CompleteIndexVector& indexVector) const {
        return std::make_unique<
          DecomposableDynamicPartialRuleEvaluation<DenseDecomposableStatisticVector<float64>, CompleteIndexVector>>(
          indexVector, threshold_, exponent_, l1RegularizationWeight_, l2RegularizationWeight_);
    }

    std::unique_ptr<IRuleEvaluation<DenseDecomposableStatisticVector<float64>>>
      DecomposableDynamicPartialRuleEvaluationFactory::create(
        const DenseDecomposableStatisticVector<float64>& statisticVector, const PartialIndexVector& indexVector) const {
        return std::make_unique<
          DecomposableCompleteRuleEvaluation<DenseDecomposableStatisticVector<float64>, PartialIndexVector>>(
          indexVector, l1RegularizationWeight_, l2RegularizationWeight_);
    }

    std::unique_ptr<IRuleEvaluation<SparseDecomposableStatisticVector<float32, uint32>>>
      DecomposableDynamicPartialRuleEvaluationFactory::create(
        const SparseDecomposableStatisticVector<float32, uint32>& statisticVector,
        const CompleteIndexVector& indexVector) const {
        return std::make_unique<DecomposableDynamicPartialRuleEvaluation<
          SparseDecomposableStatisticVector<float32, uint32>, CompleteIndexVector>>(
          indexVector, threshold_, exponent_, l1RegularizationWeight_, l2RegularizationWeight_);
    }

    std::unique_ptr<IRuleEvaluation<SparseDecomposableStatisticVector<float32, uint32>>>
      DecomposableDynamicPartialRuleEvaluationFactory::create(
        const SparseDecomposableStatisticVector<float32, uint32>& statisticVector,
        const PartialIndexVector& indexVector) const {
        return std::make_unique<
          DecomposableCompleteRuleEvaluation<SparseDecomposableStatisticVector<float32, uint32>, PartialIndexVector>>(
          indexVector, l1RegularizationWeight_, l2RegularizationWeight_);
    }

    std::unique_ptr<IRuleEvaluation<SparseDecomposableStatisticVector<float32, float32>>>
      DecomposableDynamicPartialRuleEvaluationFactory::create(
        const SparseDecomposableStatisticVector<float32, float32>& statisticVector,
        const CompleteIndexVector& indexVector) const {
        return std::make_unique<DecomposableDynamicPartialRuleEvaluation<
          SparseDecomposableStatisticVector<float32, float32>, CompleteIndexVector>>(
          indexVector, threshold_, exponent_, l1RegularizationWeight_, l2RegularizationWeight_);
    }

    std::unique_ptr<IRuleEvaluation<SparseDecomposableStatisticVector<float32, float32>>>
      DecomposableDynamicPartialRuleEvaluationFactory::create(
        const SparseDecomposableStatisticVector<float32, float32>& statisticVector,
        const PartialIndexVector& indexVector) const {
        return std::make_unique<
          DecomposableCompleteRuleEvaluation<SparseDecomposableStatisticVector<float32, float32>, PartialIndexVector>>(
          indexVector, l1RegularizationWeight_, l2RegularizationWeight_);
    }

    std::unique_ptr<IRuleEvaluation<SparseDecomposableStatisticVector<float64, uint32>>>
      DecomposableDynamicPartialRuleEvaluationFactory::create(
        const SparseDecomposableStatisticVector<float64, uint32>& statisticVector,
        const CompleteIndexVector& indexVector) const {
        return std::make_unique<DecomposableDynamicPartialRuleEvaluation<
          SparseDecomposableStatisticVector<float64, uint32>, CompleteIndexVector>>(
          indexVector, threshold_, exponent_, l1RegularizationWeight_, l2RegularizationWeight_);
    }

    std::unique_ptr<IRuleEvaluation<SparseDecomposableStatisticVector<float64, uint32>>>
      DecomposableDynamicPartialRuleEvaluationFactory::create(
        const SparseDecomposableStatisticVector<float64, uint32>& statisticVector,
        const PartialIndexVector& indexVector) const {
        return std::make_unique<
          DecomposableCompleteRuleEvaluation<SparseDecomposableStatisticVector<float64, uint32>, PartialIndexVector>>(
          indexVector, l1RegularizationWeight_, l2RegularizationWeight_);
    }

    std::unique_ptr<IRuleEvaluation<SparseDecomposableStatisticVector<float64, float32>>>
      DecomposableDynamicPartialRuleEvaluationFactory::create(
        const SparseDecomposableStatisticVector<float64, float32>& statisticVector,
        const CompleteIndexVector& indexVector) const {
        return std::make_unique<DecomposableDynamicPartialRuleEvaluation<
          SparseDecomposableStatisticVector<float64, float32>, CompleteIndexVector>>(
          indexVector, threshold_, exponent_, l1RegularizationWeight_, l2RegularizationWeight_);
    }

    std::unique_ptr<IRuleEvaluation<SparseDecomposableStatisticVector<float64, float32>>>
      DecomposableDynamicPartialRuleEvaluationFactory::create(
        const SparseDecomposableStatisticVector<float64, float32>& statisticVector,
        const PartialIndexVector& indexVector) const {
        return std::make_unique<
          DecomposableCompleteRuleEvaluation<SparseDecomposableStatisticVector<float64, float32>, PartialIndexVector>>(
          indexVector, l1RegularizationWeight_, l2RegularizationWeight_);
    }

}
