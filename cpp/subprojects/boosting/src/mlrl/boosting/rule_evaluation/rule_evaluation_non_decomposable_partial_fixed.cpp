#include "mlrl/boosting/rule_evaluation/rule_evaluation_non_decomposable_partial_fixed.hpp"

#include "rule_evaluation_non_decomposable_complete_common.hpp"
#include "rule_evaluation_non_decomposable_partial_common.hpp"
#include "rule_evaluation_non_decomposable_partial_fixed_common.hpp"

namespace boosting {

    /**
     * Allows to calculate the predictions of partial rules that predict for a predefined number of outputs, as well as
     * their overall quality, based on the gradients and Hessians that are stored by a
     * `DenseNonDecomposableStatisticVector` using L1 and L2 regularization.
     *
     * @tparam StatisticVector  The type of the vector that provides access to the gradients and Hessians
     * @tparam IndexVector      The type of the vector that provides access to the indices of the outputs for which
     *                          predictions should be calculated
     */
    template<typename StatisticVector, typename IndexVector>
    class DenseNonDecomposableFixedPartialRuleEvaluation final
        : public AbstractNonDecomposableRuleEvaluation<StatisticVector, IndexVector> {
        private:

            const IndexVector& outputIndices_;

            PartialIndexVector indexVector_;

            DenseScoreVector<PartialIndexVector> scoreVector_;

            const float32 l1RegularizationWeight_;

            const float32 l2RegularizationWeight_;

            const std::unique_ptr<Blas<typename StatisticVector::statistic_type>> blasPtr_;

            const std::unique_ptr<Lapack<typename StatisticVector::statistic_type>> lapackPtr_;

            SparseArrayVector<float64> tmpVector_;

        public:

            /**
             * @param outputIndices             A reference to an object of template type `IndexVector` that provides
             *                                  access to the indices of the outputs for which the rules may predict
             * @param numPredictions            The number of outputs for which the rules should predict
             * @param l1RegularizationWeight    The weight of the L1 regularization that is applied for calculating the
             *                                  scores to be predicted by rules
             * @param l2RegularizationWeight    The weight of the L2 regularization that is applied for calculating the
             *                                  scores to be predicted by rules
             * @param blasPtr                   An unique pointer to an object of type `Blas` that allows to execute
             *                                  BLAS routines
             * @param lapackPtr                 An unique pointer to an object of type `Lapack` that allows to execute
             *                                  LAPACK routines
             */
            DenseNonDecomposableFixedPartialRuleEvaluation(
              const IndexVector& outputIndices, uint32 numPredictions, float32 l1RegularizationWeight,
              float32 l2RegularizationWeight, std::unique_ptr<Blas<typename StatisticVector::statistic_type>> blasPtr,
              std::unique_ptr<Lapack<typename StatisticVector::statistic_type>> lapackPtr)
                : AbstractNonDecomposableRuleEvaluation<StatisticVector, IndexVector>(numPredictions, *lapackPtr),
                  outputIndices_(outputIndices), indexVector_(numPredictions), scoreVector_(indexVector_, false),
                  l1RegularizationWeight_(l1RegularizationWeight), l2RegularizationWeight_(l2RegularizationWeight),
                  blasPtr_(std::move(blasPtr)), lapackPtr_(std::move(lapackPtr)),
                  tmpVector_(outputIndices.getNumElements()) {}

            /**
             * @see `IRuleEvaluation::evaluate`
             */
            const IScoreVector& calculateScores(StatisticVector& statisticVector) override {
                uint32 numOutputs = statisticVector.getNumGradients();
                uint32 numPredictions = indexVector_.getNumElements();
                typename StatisticVector::gradient_const_iterator gradientIterator = statisticVector.gradients_cbegin();
                typename StatisticVector::hessian_diagonal_const_iterator hessianIterator =
                  statisticVector.hessians_diagonal_cbegin();
                SparseArrayVector<float64>::iterator tmpIterator = tmpVector_.begin();
                sortOutputWiseCriteria(tmpIterator, gradientIterator, hessianIterator, numOutputs, numPredictions,
                                       l1RegularizationWeight_, l2RegularizationWeight_);

                // Copy gradients to the vector of ordinates and add the L1 regularization weight...
                PartialIndexVector::iterator indexIterator = indexVector_.begin();
                typename DenseScoreVector<IndexVector>::value_iterator valueIterator = scoreVector_.values_begin();
                typename IndexVector::const_iterator outputIndexIterator = outputIndices_.cbegin();

                for (uint32 i = 0; i < numPredictions; i++) {
                    const IndexedValue<float64>& entry = tmpIterator[i];
                    uint32 index = entry.index;
                    indexIterator[i] = outputIndexIterator[index];
                    valueIterator[i] = -gradientIterator[index];
                }

                addL1RegularizationWeight<float64>(valueIterator, numPredictions, l1RegularizationWeight_);

                // Copy Hessians to the matrix of coefficients and add the L2 regularization weight to its diagonal...
                copyCoefficients<float64, PartialIndexVector::iterator>(
                  statisticVector.hessians_cbegin(), indexIterator, this->sysvTmpArray1_.begin(), numPredictions);
                addL2RegularizationWeight<float64>(this->sysvTmpArray1_.begin(), numPredictions,
                                                   l2RegularizationWeight_);

                // Calculate the scores to be predicted for individual outputs by solving a system of linear
                // equations...
                lapackPtr_->sysv(this->sysvTmpArray1_.begin(), this->sysvTmpArray2_.begin(),
                                 this->sysvTmpArray3_.begin(), valueIterator, numPredictions, this->sysvLwork_);

                // Calculate the overall quality...
                float64 quality = calculateOverallQuality<float64>(
                  valueIterator, statisticVector.gradients_begin(), statisticVector.hessians_begin(),
                  this->spmvTmpArray_.begin(), numPredictions, *blasPtr_);

                // Evaluate regularization term...
                quality += calculateRegularizationTerm<float64>(valueIterator, numPredictions, l1RegularizationWeight_,
                                                                l2RegularizationWeight_);

                scoreVector_.quality = quality;
                return scoreVector_;
            }
    };

    NonDecomposableFixedPartialRuleEvaluationFactory::NonDecomposableFixedPartialRuleEvaluationFactory(
      float32 outputRatio, uint32 minOutputs, uint32 maxOutputs, float32 l1RegularizationWeight,
      float32 l2RegularizationWeight, const BlasFactory& blasFactory, const LapackFactory& lapackFactory)
        : outputRatio_(outputRatio), minOutputs_(minOutputs), maxOutputs_(maxOutputs),
          l1RegularizationWeight_(l1RegularizationWeight), l2RegularizationWeight_(l2RegularizationWeight),
          blasFactory_(blasFactory), lapackFactory_(lapackFactory) {}

    std::unique_ptr<IRuleEvaluation<DenseNonDecomposableStatisticVector<float64>>>
      NonDecomposableFixedPartialRuleEvaluationFactory::create(
        const DenseNonDecomposableStatisticVector<float64>& statisticVector,
        const CompleteIndexVector& indexVector) const {
        uint32 numPredictions =
          util::calculateBoundedFraction(indexVector.getNumElements(), outputRatio_, minOutputs_, maxOutputs_);
        return std::make_unique<DenseNonDecomposableFixedPartialRuleEvaluation<
          DenseNonDecomposableStatisticVector<float64>, CompleteIndexVector>>(
          indexVector, numPredictions, l1RegularizationWeight_, l2RegularizationWeight_, blasFactory_.create64Bit(),
          lapackFactory_.create64Bit());
    }

    std::unique_ptr<IRuleEvaluation<DenseNonDecomposableStatisticVector<float64>>>
      NonDecomposableFixedPartialRuleEvaluationFactory::create(
        const DenseNonDecomposableStatisticVector<float64>& statisticVector,
        const PartialIndexVector& indexVector) const {
        return std::make_unique<
          DenseNonDecomposableCompleteRuleEvaluation<DenseNonDecomposableStatisticVector<float64>, PartialIndexVector>>(
          indexVector, l1RegularizationWeight_, l2RegularizationWeight_, blasFactory_.create64Bit(),
          lapackFactory_.create64Bit());
    }

}
