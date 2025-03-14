#include "mlrl/boosting/rule_evaluation/rule_evaluation_non_decomposable_partial_dynamic.hpp"

#include "rule_evaluation_non_decomposable_complete_common.hpp"
#include "rule_evaluation_non_decomposable_partial_common.hpp"
#include "rule_evaluation_non_decomposable_partial_dynamic_common.hpp"

namespace boosting {

    /**
     * Allows to calculate the predictions of partial rules that predict for a subset of the available outputs that is
     * determined dynamically, as well as their overall quality, based on the gradients and Hessians that are stored by
     * a `DenseNonDecomposableStatisticVector` using L1 and L2 regularization.
     *
     * @tparam StatisticVector  The type of the vector that provides access to the gradients and Hessians
     * @tparam IndexVector      The type of the vector that provides access to the indices of the outputs for which
     *                          predictions should be calculated
     */
    template<typename StatisticVector, typename IndexVector>
    class DenseNonDecomposableDynamicPartialRuleEvaluation final
        : public AbstractNonDecomposableRuleEvaluation<StatisticVector, IndexVector> {
        private:

            typedef typename StatisticVector::statistic_type statistic_type;

            const IndexVector& outputIndices_;

            PartialIndexVector indexVector_;

            DenseScoreVector<statistic_type, PartialIndexVector> scoreVector_;

            const float32 threshold_;

            const float32 exponent_;

            const float32 l1RegularizationWeight_;

            const float32 l2RegularizationWeight_;

            const std::unique_ptr<Blas<statistic_type>> blasPtr_;

            const std::unique_ptr<Lapack<statistic_type>> lapackPtr_;

        public:

            /**
             * @param outputIndices             A reference to an object of template type `IndexVector` that provides
             *                                  access to the indices of the outputs for which the rules may predict
             * @param threshold                 A threshold that affects for how many outputs the rule heads should
             *                                  predict
             * @param exponent                  An exponent that is used to weigh the estimated predictive quality for
             *                                  individual ouputs
             * @param l1RegularizationWeight    The weight of the L1 regularization that is applied for calculating the
             *                                  scores to be predicted by rules
             * @param l2RegularizationWeight    The weight of the L2 regularization that is applied for calculating the
             *                                  scores to be predicted by rules
             * @param blasPtr                   An unique pointer to an object of type `Blas` that allows to execute
             *                                  BLAS routines
             * @param lapackPtr                 An unique pointer to an object of type `Lapack` that allows to execute
             *                                  LAPACK routines
             */
            DenseNonDecomposableDynamicPartialRuleEvaluation(const IndexVector& outputIndices, float32 threshold,
                                                             float32 exponent, float32 l1RegularizationWeight,
                                                             float32 l2RegularizationWeight,
                                                             std::unique_ptr<Blas<statistic_type>> blasPtr,
                                                             std::unique_ptr<Lapack<statistic_type>> lapackPtr)
                : AbstractNonDecomposableRuleEvaluation<StatisticVector, IndexVector>(outputIndices.getNumElements(),
                                                                                      *lapackPtr),
                  outputIndices_(outputIndices), indexVector_(outputIndices.getNumElements()),
                  scoreVector_(indexVector_, true), threshold_(1.0f - threshold), exponent_(exponent),
                  l1RegularizationWeight_(l1RegularizationWeight), l2RegularizationWeight_(l2RegularizationWeight),
                  blasPtr_(std::move(blasPtr)), lapackPtr_(std::move(lapackPtr)) {}

            /**
             * @see `IRuleEvaluation::evaluate`
             */
            const IScoreVector& calculateScores(StatisticVector& statisticVector) override {
                uint32 numOutputs = statisticVector.getNumGradients();
                typename StatisticVector::gradient_const_iterator gradientIterator = statisticVector.gradients_cbegin();
                typename StatisticVector::hessian_diagonal_const_iterator hessianIterator =
                  statisticVector.hessians_diagonal_cbegin();
                typename DenseScoreVector<statistic_type, IndexVector>::value_iterator valueIterator =
                  scoreVector_.values_begin();
                const std::pair<statistic_type, statistic_type> pair =
                  getMinAndMaxScore<statistic_type, typename StatisticVector::gradient_const_iterator,
                                    typename StatisticVector::hessian_diagonal_const_iterator>(
                    valueIterator, gradientIterator, hessianIterator, numOutputs, l1RegularizationWeight_,
                    l2RegularizationWeight_);
                statistic_type minAbsScore = pair.first;

                // Copy gradients to the vector of ordinates and add the L1 regularization weight...
                statistic_type threshold = calculateThreshold(minAbsScore, pair.second, threshold_, exponent_);
                PartialIndexVector::iterator indexIterator = indexVector_.begin();
                typename IndexVector::const_iterator outputIndexIterator = outputIndices_.cbegin();
                uint32 n = 0;

                for (uint32 i = 0; i < numOutputs; i++) {
                    statistic_type score = valueIterator[i];

                    if (calculateWeightedScore(score, minAbsScore, exponent_) >= threshold) {
                        indexIterator[n] = outputIndexIterator[i];
                        valueIterator[n] = -gradientIterator[i];
                        n++;
                    }
                }

                indexVector_.setNumElements(n, false);
                addL1RegularizationWeight(valueIterator, n, l1RegularizationWeight_);

                // Copy Hessians to the matrix of coefficients and add the L2 regularization weight to its diagonal...
                copyCoefficients(statisticVector.hessians_cbegin(), indexIterator, this->sysvTmpArray1_.begin(), n);
                addL2RegularizationWeight(this->sysvTmpArray1_.begin(), n, l2RegularizationWeight_);

                // Calculate the scores to be predicted for individual outputs by solving a system of linear
                // equations...
                lapackPtr_->sysv(this->sysvTmpArray1_.begin(), this->sysvTmpArray2_.begin(),
                                 this->sysvTmpArray3_.begin(), valueIterator, n, this->sysvLwork_);

                // Calculate the overall quality...
                statistic_type quality =
                  calculateOverallQuality(valueIterator, statisticVector.gradients_begin(),
                                          statisticVector.hessians_begin(), this->spmvTmpArray_.begin(), n, *blasPtr_);

                // Evaluate regularization term...
                quality +=
                  calculateRegularizationTerm(valueIterator, n, l1RegularizationWeight_, l2RegularizationWeight_);

                scoreVector_.quality = quality;
                return scoreVector_;
            }
    };

    NonDecomposableDynamicPartialRuleEvaluationFactory::NonDecomposableDynamicPartialRuleEvaluationFactory(
      float32 threshold, float32 exponent, float32 l1RegularizationWeight, float32 l2RegularizationWeight,
      const BlasFactory& blasFactory, const LapackFactory& lapackFactory)
        : threshold_(threshold), exponent_(exponent), l1RegularizationWeight_(l1RegularizationWeight),
          l2RegularizationWeight_(l2RegularizationWeight), blasFactory_(blasFactory), lapackFactory_(lapackFactory) {}

    std::unique_ptr<IRuleEvaluation<DenseNonDecomposableStatisticVector<float32>>>
      NonDecomposableDynamicPartialRuleEvaluationFactory::create(
        const DenseNonDecomposableStatisticVector<float32>& statisticVector,
        const CompleteIndexVector& indexVector) const {
        return std::make_unique<DenseNonDecomposableDynamicPartialRuleEvaluation<
          DenseNonDecomposableStatisticVector<float32>, CompleteIndexVector>>(
          indexVector, threshold_, exponent_, l1RegularizationWeight_, l2RegularizationWeight_,
          blasFactory_.create32Bit(), lapackFactory_.create32Bit());
    }

    std::unique_ptr<IRuleEvaluation<DenseNonDecomposableStatisticVector<float32>>>
      NonDecomposableDynamicPartialRuleEvaluationFactory::create(
        const DenseNonDecomposableStatisticVector<float32>& statisticVector,
        const PartialIndexVector& indexVector) const {
        return std::make_unique<
          DenseNonDecomposableCompleteRuleEvaluation<DenseNonDecomposableStatisticVector<float32>, PartialIndexVector>>(
          indexVector, l1RegularizationWeight_, l2RegularizationWeight_, blasFactory_.create32Bit(),
          lapackFactory_.create32Bit());
    }

    std::unique_ptr<IRuleEvaluation<DenseNonDecomposableStatisticVector<float64>>>
      NonDecomposableDynamicPartialRuleEvaluationFactory::create(
        const DenseNonDecomposableStatisticVector<float64>& statisticVector,
        const CompleteIndexVector& indexVector) const {
        return std::make_unique<DenseNonDecomposableDynamicPartialRuleEvaluation<
          DenseNonDecomposableStatisticVector<float64>, CompleteIndexVector>>(
          indexVector, threshold_, exponent_, l1RegularizationWeight_, l2RegularizationWeight_,
          blasFactory_.create64Bit(), lapackFactory_.create64Bit());
    }

    std::unique_ptr<IRuleEvaluation<DenseNonDecomposableStatisticVector<float64>>>
      NonDecomposableDynamicPartialRuleEvaluationFactory::create(
        const DenseNonDecomposableStatisticVector<float64>& statisticVector,
        const PartialIndexVector& indexVector) const {
        return std::make_unique<
          DenseNonDecomposableCompleteRuleEvaluation<DenseNonDecomposableStatisticVector<float64>, PartialIndexVector>>(
          indexVector, l1RegularizationWeight_, l2RegularizationWeight_, blasFactory_.create64Bit(),
          lapackFactory_.create64Bit());
    }

}
