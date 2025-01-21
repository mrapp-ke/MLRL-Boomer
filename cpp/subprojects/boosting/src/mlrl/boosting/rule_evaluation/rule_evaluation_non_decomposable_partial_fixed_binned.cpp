#include "mlrl/boosting/rule_evaluation/rule_evaluation_non_decomposable_partial_fixed_binned.hpp"

#include "rule_evaluation_non_decomposable_binned_common.hpp"
#include "rule_evaluation_non_decomposable_partial_fixed_common.hpp"

namespace boosting {

    /**
     * Allows to calculate the predictions of partial rules that predict for a predefined number of labels, as well as
     * their overall quality, based on the gradients and Hessians that are stored by a
     * `DenseNonDecomposableStatisticVector` using L1 and L2 regularization. The labels are assigned to bins based on
     * the gradients and Hessians.
     *
     * @tparam StatisticVector  The type of the vector that provides access to the gradients and Hessians
     * @tparam IndexVector      The type of the vector that provides access to the indices of the labels for which
     *                          predictions should be calculated
     */
    template<typename StatisticVector, typename IndexVector>
    class DenseNonDecomposableFixedPartialBinnedRuleEvaluation final
        : public AbstractNonDecomposableBinnedRuleEvaluation<StatisticVector, PartialIndexVector> {
        private:

            const IndexVector& labelIndices_;

            const std::unique_ptr<PartialIndexVector> indexVectorPtr_;

            SparseArrayVector<float64> tmpVector_;

        protected:

            uint32 calculateOutputWiseCriteria(const StatisticVector& statisticVector, float64* criteria,
                                               uint32 numCriteria, float32 l1RegularizationWeight,
                                               float32 l2RegularizationWeight) override {
                uint32 numOutputs = statisticVector.getNumGradients();
                uint32 numPredictions = indexVectorPtr_->getNumElements();
                typename StatisticVector::gradient_const_iterator gradientIterator = statisticVector.gradients_cbegin();
                typename StatisticVector::hessian_diagonal_const_iterator hessianIterator =
                  statisticVector.hessians_diagonal_cbegin();
                SparseArrayVector<float64>::iterator tmpIterator = tmpVector_.begin();
                sortOutputWiseCriteria(tmpIterator, gradientIterator, hessianIterator, numOutputs, numPredictions,
                                       l1RegularizationWeight, l2RegularizationWeight);
                PartialIndexVector::iterator indexIterator = indexVectorPtr_->begin();
                typename IndexVector::const_iterator labelIndexIterator = labelIndices_.cbegin();

                for (uint32 i = 0; i < numCriteria; i++) {
                    const IndexedValue<float64>& entry = tmpIterator[i];
                    indexIterator[i] = labelIndexIterator[entry.index];
                    criteria[i] = entry.value;
                }

                return numCriteria;
            }

        public:

            /**
             * @param labelIndices              A reference to an object of template type `IndexVector` that provides
             *                                  access to the indices of the labels for which the rules may predict
             * @param maxBins                   The maximum number of bins
             * @param indexVectorPtr            An unique pointer to an object of type `PartialIndexVector` that stores
             *                                  the indices of the labels for which a rule predicts
             * @param l1RegularizationWeight    The weight of the L1 regularization that is applied for calculating the
             *                                  scores to be predicted by rules
             * @param l2RegularizationWeight    The weight of the L2 regularization that is applied for calculating the
             *                                  scores to be predicted by rules
             * @param binningPtr                An unique pointer to an object of type `ILabelBinning` that should be
             *                                  used to assign labels to bins
             * @param blas                      A reference to an object of type `Blas` that allows to execute BLAS
             *                                  routines
             * @param lapack                    A reference to an object of type `Lapack` that allows to execute LAPACK
             *                                  routines
             */
            DenseNonDecomposableFixedPartialBinnedRuleEvaluation(const IndexVector& labelIndices, uint32 maxBins,
                                                                 std::unique_ptr<PartialIndexVector> indexVectorPtr,
                                                                 float32 l1RegularizationWeight,
                                                                 float32 l2RegularizationWeight,
                                                                 std::unique_ptr<ILabelBinning> binningPtr,
                                                                 const Blas& blas, const Lapack& lapack)
                : AbstractNonDecomposableBinnedRuleEvaluation<StatisticVector, PartialIndexVector>(
                    *indexVectorPtr, false, maxBins, l1RegularizationWeight, l2RegularizationWeight,
                    std::move(binningPtr), blas, lapack),
                  labelIndices_(labelIndices), indexVectorPtr_(std::move(indexVectorPtr)),
                  tmpVector_(labelIndices.getNumElements()) {}
    };

    NonDecomposableFixedPartialBinnedRuleEvaluationFactory::NonDecomposableFixedPartialBinnedRuleEvaluationFactory(
      float32 labelRatio, uint32 minLabels, uint32 maxLabels, float32 l1RegularizationWeight,
      float32 l2RegularizationWeight, std::unique_ptr<ILabelBinningFactory> labelBinningFactoryPtr, const Blas& blas,
      const Lapack& lapack)
        : labelRatio_(labelRatio), minLabels_(minLabels), maxLabels_(maxLabels),
          l1RegularizationWeight_(l1RegularizationWeight), l2RegularizationWeight_(l2RegularizationWeight),
          labelBinningFactoryPtr_(std::move(labelBinningFactoryPtr)), blas_(blas), lapack_(lapack) {}

    std::unique_ptr<IRuleEvaluation<DenseNonDecomposableStatisticVector<float64>>>
      NonDecomposableFixedPartialBinnedRuleEvaluationFactory::create(
        const DenseNonDecomposableStatisticVector<float64>& statisticVector,
        const CompleteIndexVector& indexVector) const {
        uint32 numPredictions =
          util::calculateBoundedFraction(statisticVector.getNumGradients(), labelRatio_, minLabels_, maxLabels_);
        std::unique_ptr<PartialIndexVector> indexVectorPtr = std::make_unique<PartialIndexVector>(numPredictions);
        std::unique_ptr<ILabelBinning> labelBinningPtr = labelBinningFactoryPtr_->create();
        uint32 maxBins = labelBinningPtr->getMaxBins(numPredictions);
        return std::make_unique<DenseNonDecomposableFixedPartialBinnedRuleEvaluation<
          DenseNonDecomposableStatisticVector<float64>, CompleteIndexVector>>(
          indexVector, maxBins, std::move(indexVectorPtr), l1RegularizationWeight_, l2RegularizationWeight_,
          std::move(labelBinningPtr), blas_, lapack_);
    }

    std::unique_ptr<IRuleEvaluation<DenseNonDecomposableStatisticVector<float64>>>
      NonDecomposableFixedPartialBinnedRuleEvaluationFactory::create(
        const DenseNonDecomposableStatisticVector<float64>& statisticVector,
        const PartialIndexVector& indexVector) const {
        std::unique_ptr<ILabelBinning> labelBinningPtr = labelBinningFactoryPtr_->create();
        uint32 maxBins = labelBinningPtr->getMaxBins(indexVector.getNumElements());
        return std::make_unique<DenseNonDecomposableCompleteBinnedRuleEvaluation<PartialIndexVector>>(
          indexVector, maxBins, l1RegularizationWeight_, l2RegularizationWeight_, std::move(labelBinningPtr), blas_,
          lapack_);
    }

}
