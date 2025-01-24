#include "mlrl/boosting/rule_evaluation/rule_evaluation_decomposable_partial_fixed.hpp"

#include "rule_evaluation_decomposable_complete_common.hpp"
#include "rule_evaluation_decomposable_partial_fixed_common.hpp"

namespace boosting {

    /**
     * Allows to calculate the predictions of partial rules that predict for a predefined number of outputs, as well as
     * their overall quality, based on the gradients and Hessians that are stored by a vector using L1 and L2
     * regularization.
     *
     * @tparam StatisticVector  The type of the vector that provides access to the gradients and Hessians
     * @tparam IndexVector      The type of the vector that provides access to the indices of the outputs for which
     *                          predictions should be calculated
     */
    template<typename StatisticVector, typename IndexVector>
    class DecomposableFixedPartialRuleEvaluation final : public IRuleEvaluation<StatisticVector> {
        private:

            const IndexVector& outputIndices_;

            PartialIndexVector indexVector_;

            DenseScoreVector<typename StatisticVector::statistic_type, PartialIndexVector> scoreVector_;

            const float32 l1RegularizationWeight_;

            const float32 l2RegularizationWeight_;

            SparseArrayVector<typename StatisticVector::statistic_type> tmpVector_;

        public:

            /**
             * @param outputIndices             A reference to an object of template type `IndexVector` that provides
             *                                  access to the indices of the outputs for which the rules may predict
             * @param numPredictions            The number of outputs for which the rules should predict
             * @param l1RegularizationWeight    The weight of the L1 regularization that is applied for calculating the
             *                                  scores to be predicted by rules
             * @param l2RegularizationWeight    The weight of the L2 regularization that is applied for calculating the
             *                                  scores to be predicted by rules
             */
            DecomposableFixedPartialRuleEvaluation(const IndexVector& outputIndices, uint32 numPredictions,
                                                   float32 l1RegularizationWeight, float32 l2RegularizationWeight)
                : outputIndices_(outputIndices), indexVector_(numPredictions), scoreVector_(indexVector_, false),
                  l1RegularizationWeight_(l1RegularizationWeight), l2RegularizationWeight_(l2RegularizationWeight),
                  tmpVector_(outputIndices.getNumElements()) {}

            const IScoreVector& calculateScores(StatisticVector& statisticVector) override {
                uint32 numElements = statisticVector.getNumElements();
                uint32 numPredictions = indexVector_.getNumElements();
                typename StatisticVector::const_iterator statisticIterator = statisticVector.cbegin();
                typename SparseArrayVector<typename StatisticVector::statistic_type>::iterator tmpIterator =
                  tmpVector_.begin();
                sortOutputWiseScores(tmpIterator, statisticIterator, numElements, numPredictions,
                                     l1RegularizationWeight_, l2RegularizationWeight_);
                PartialIndexVector::iterator indexIterator = indexVector_.begin();
                typename DenseScoreVector<typename StatisticVector::statistic_type, PartialIndexVector>::value_iterator
                  valueIterator = scoreVector_.values_begin();
                typename IndexVector::const_iterator outputIndexIterator = outputIndices_.cbegin();
                typename StatisticVector::statistic_type quality = 0;

                for (uint32 i = 0; i < numPredictions; i++) {
                    const IndexedValue<typename StatisticVector::statistic_type>& entry = tmpIterator[i];
                    uint32 index = entry.index;
                    typename StatisticVector::statistic_type predictedScore = entry.value;
                    indexIterator[i] = outputIndexIterator[index];
                    valueIterator[i] = predictedScore;
                    const typename StatisticVector::value_type& statistic = statisticIterator[index];
                    quality += calculateOutputWiseQuality(predictedScore, statistic.gradient, statistic.hessian,
                                                          l1RegularizationWeight_, l2RegularizationWeight_);
                }

                scoreVector_.quality = quality;
                return scoreVector_;
            }
    };

    DecomposableFixedPartialRuleEvaluationFactory::DecomposableFixedPartialRuleEvaluationFactory(
      float32 outputRatio, uint32 minOutputs, uint32 maxOutputs, float32 l1RegularizationWeight,
      float32 l2RegularizationWeight)
        : outputRatio_(outputRatio), minOutputs_(minOutputs), maxOutputs_(maxOutputs),
          l1RegularizationWeight_(l1RegularizationWeight), l2RegularizationWeight_(l2RegularizationWeight) {}

    std::unique_ptr<IRuleEvaluation<DenseDecomposableStatisticVector<float64>>>
      DecomposableFixedPartialRuleEvaluationFactory::create(
        const DenseDecomposableStatisticVector<float64>& statisticVector,
        const CompleteIndexVector& indexVector) const {
        uint32 numPredictions =
          util::calculateBoundedFraction(indexVector.getNumElements(), outputRatio_, minOutputs_, maxOutputs_);
        return std::make_unique<
          DecomposableFixedPartialRuleEvaluation<DenseDecomposableStatisticVector<float64>, CompleteIndexVector>>(
          indexVector, numPredictions, l1RegularizationWeight_, l2RegularizationWeight_);
    }

    std::unique_ptr<IRuleEvaluation<DenseDecomposableStatisticVector<float64>>>
      DecomposableFixedPartialRuleEvaluationFactory::create(
        const DenseDecomposableStatisticVector<float64>& statisticVector, const PartialIndexVector& indexVector) const {
        return std::make_unique<
          DecomposableCompleteRuleEvaluation<DenseDecomposableStatisticVector<float64>, PartialIndexVector>>(
          indexVector, l1RegularizationWeight_, l2RegularizationWeight_);
    }

    std::unique_ptr<IRuleEvaluation<SparseDecomposableStatisticVector<float64, uint32>>>
      DecomposableFixedPartialRuleEvaluationFactory::create(
        const SparseDecomposableStatisticVector<float64, uint32>& statisticVector,
        const CompleteIndexVector& indexVector) const {
        uint32 numPredictions =
          util::calculateBoundedFraction(indexVector.getNumElements(), outputRatio_, minOutputs_, maxOutputs_);
        return std::make_unique<DecomposableFixedPartialRuleEvaluation<
          SparseDecomposableStatisticVector<float64, uint32>, CompleteIndexVector>>(
          indexVector, numPredictions, l1RegularizationWeight_, l2RegularizationWeight_);
    }

    std::unique_ptr<IRuleEvaluation<SparseDecomposableStatisticVector<float64, uint32>>>
      DecomposableFixedPartialRuleEvaluationFactory::create(
        const SparseDecomposableStatisticVector<float64, uint32>& statisticVector,
        const PartialIndexVector& indexVector) const {
        return std::make_unique<
          DecomposableCompleteRuleEvaluation<SparseDecomposableStatisticVector<float64, uint32>, PartialIndexVector>>(
          indexVector, l1RegularizationWeight_, l2RegularizationWeight_);
    }

    std::unique_ptr<IRuleEvaluation<SparseDecomposableStatisticVector<float64, float32>>>
      DecomposableFixedPartialRuleEvaluationFactory::create(
        const SparseDecomposableStatisticVector<float64, float32>& statisticVector,
        const CompleteIndexVector& indexVector) const {
        uint32 numPredictions =
          util::calculateBoundedFraction(indexVector.getNumElements(), outputRatio_, minOutputs_, maxOutputs_);
        return std::make_unique<DecomposableFixedPartialRuleEvaluation<
          SparseDecomposableStatisticVector<float64, float32>, CompleteIndexVector>>(
          indexVector, numPredictions, l1RegularizationWeight_, l2RegularizationWeight_);
    }

    std::unique_ptr<IRuleEvaluation<SparseDecomposableStatisticVector<float64, float32>>>
      DecomposableFixedPartialRuleEvaluationFactory::create(
        const SparseDecomposableStatisticVector<float64, float32>& statisticVector,
        const PartialIndexVector& indexVector) const {
        return std::make_unique<
          DecomposableCompleteRuleEvaluation<SparseDecomposableStatisticVector<float64, float32>, PartialIndexVector>>(
          indexVector, l1RegularizationWeight_, l2RegularizationWeight_);
    }

}
