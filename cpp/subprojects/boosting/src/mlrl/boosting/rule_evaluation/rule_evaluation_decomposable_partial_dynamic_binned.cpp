#include "mlrl/boosting/rule_evaluation/rule_evaluation_decomposable_partial_dynamic_binned.hpp"

#include "rule_evaluation_decomposable_binned_common.hpp"
#include "rule_evaluation_decomposable_partial_dynamic_common.hpp"

namespace boosting {

    /**
     * Allows to calculate the predictions of partial rules that predict for a subset of the available labels that is
     * determined dynamically, as well as their overall quality, based on the gradients and Hessians that are stored by
     * a vector using L1 and L2 regularization. The labels are assigned to bins based on the gradients and Hessians.
     *
     * @tparam StatisticVector  The type of the vector that provides access to the gradients and Hessians
     * @tparam IndexVector      The type of the vector that provides access to the indices of the labels for which
     *                          predictions should be calculated
     */
    template<typename StatisticVector, typename IndexVector>
    class DecomposableDynamicPartialBinnedRuleEvaluation final
        : public AbstractDecomposableBinnedRuleEvaluation<StatisticVector, PartialIndexVector> {
        private:

            typedef typename StatisticVector::statistic_type statistic_type;

            const IndexVector& labelIndices_;

            const std::unique_ptr<PartialIndexVector> indexVectorPtr_;

            const float32 threshold_;

            const float32 exponent_;

        protected:

            uint32 calculateOutputWiseCriteria(const StatisticVector& statisticVector,
                                               typename View<statistic_type>::iterator criteria, uint32 numCriteria,
                                               float32 l1RegularizationWeight,
                                               float32 l2RegularizationWeight) override {
                uint32 numElements = statisticVector.getNumElements();
                typename StatisticVector::const_iterator statisticIterator = statisticVector.cbegin();
                const std::pair<statistic_type, statistic_type> pair =
                  getMinAndMaxScore(statisticIterator, numElements, l1RegularizationWeight, l2RegularizationWeight);
                statistic_type minAbsScore = pair.first;
                statistic_type threshold = calculateThreshold(minAbsScore, pair.second, threshold_, exponent_);
                PartialIndexVector::iterator indexIterator = indexVectorPtr_->begin();
                typename IndexVector::const_iterator labelIndexIterator = labelIndices_.cbegin();
                uint32 n = 0;

                for (uint32 i = 0; i < numElements; i++) {
                    const Statistic<statistic_type>& statistic = statisticIterator[i];
                    statistic_type score = calculateOutputWiseScore(statistic.gradient, statistic.hessian,
                                                                    l1RegularizationWeight, l2RegularizationWeight);

                    if (calculateWeightedScore(score, minAbsScore, exponent_) >= threshold) {
                        indexIterator[n] = labelIndexIterator[i];
                        criteria[n] = score;
                        n++;
                    }
                }

                indexVectorPtr_->setNumElements(n, false);
                return n;
            }

        public:

            /**
             * @param labelIndices              A reference to an object of template type `IndexVector` that provides
             *                                  access to the indices of the labels for which the rules may predict
             * @param indexVectorPtr            An unique pointer to an object of type `PartialIndexVector` that stores
             *                                  the indices of the labels for which a rule predicts
             * @param threshold                 A threshold that affects for how many labels the rule heads should
             *                                  predict
             * @param exponent                  An exponent that is used to weigh the estimated predictive quality for
             *                                  individual labels
             * @param l1RegularizationWeight    The weight of the L1 regularization that is applied for calculating the
             *                                  scores to be predicted by rules
             * @param l2RegularizationWeight    The weight of the L2 regularization that is applied for calculating the
             *                                  scores to be predicted by rules
             * @param binningPtr                An unique pointer to an object of type `ILabelBinning` that should be
             *                                  used to assign labels to bins
             */
            DecomposableDynamicPartialBinnedRuleEvaluation(const IndexVector& labelIndices,
                                                           std::unique_ptr<PartialIndexVector> indexVectorPtr,
                                                           float32 threshold, float32 exponent,
                                                           float32 l1RegularizationWeight,
                                                           float32 l2RegularizationWeight,
                                                           std::unique_ptr<ILabelBinning<statistic_type>> binningPtr)
                : AbstractDecomposableBinnedRuleEvaluation<StatisticVector, PartialIndexVector>(
                    *indexVectorPtr, true, l1RegularizationWeight, l2RegularizationWeight, std::move(binningPtr)),
                  labelIndices_(labelIndices), indexVectorPtr_(std::move(indexVectorPtr)), threshold_(1.0f - threshold),
                  exponent_(exponent) {}
    };

    DecomposableDynamicPartialBinnedRuleEvaluationFactory::DecomposableDynamicPartialBinnedRuleEvaluationFactory(
      float32 threshold, float32 exponent, float32 l1RegularizationWeight, float32 l2RegularizationWeight,
      std::unique_ptr<ILabelBinningFactory> labelBinningFactoryPtr)
        : threshold_(threshold), exponent_(exponent), l1RegularizationWeight_(l1RegularizationWeight),
          l2RegularizationWeight_(l2RegularizationWeight), labelBinningFactoryPtr_(std::move(labelBinningFactoryPtr)) {}

    std::unique_ptr<IRuleEvaluation<DenseDecomposableStatisticVector<float32>>>
      DecomposableDynamicPartialBinnedRuleEvaluationFactory::create(
        const DenseDecomposableStatisticVector<float32>& statisticVector,
        const CompleteIndexVector& indexVector) const {
        std::unique_ptr<PartialIndexVector> indexVectorPtr =
          std::make_unique<PartialIndexVector>(indexVector.getNumElements());
        std::unique_ptr<ILabelBinning<float32>> labelBinningPtr = labelBinningFactoryPtr_->create32Bit();
        return std::make_unique<DecomposableDynamicPartialBinnedRuleEvaluation<
          DenseDecomposableStatisticVector<float32>, CompleteIndexVector>>(
          indexVector, std::move(indexVectorPtr), threshold_, exponent_, l1RegularizationWeight_,
          l2RegularizationWeight_, std::move(labelBinningPtr));
    }

    std::unique_ptr<IRuleEvaluation<DenseDecomposableStatisticVector<float32>>>
      DecomposableDynamicPartialBinnedRuleEvaluationFactory::create(
        const DenseDecomposableStatisticVector<float32>& statisticVector, const PartialIndexVector& indexVector) const {
        std::unique_ptr<ILabelBinning<float32>> labelBinningPtr = labelBinningFactoryPtr_->create32Bit();
        return std::make_unique<
          DecomposableCompleteBinnedRuleEvaluation<DenseDecomposableStatisticVector<float32>, PartialIndexVector>>(
          indexVector, l1RegularizationWeight_, l2RegularizationWeight_, std::move(labelBinningPtr));
    }

    std::unique_ptr<IRuleEvaluation<DenseDecomposableStatisticVector<float64>>>
      DecomposableDynamicPartialBinnedRuleEvaluationFactory::create(
        const DenseDecomposableStatisticVector<float64>& statisticVector,
        const CompleteIndexVector& indexVector) const {
        std::unique_ptr<PartialIndexVector> indexVectorPtr =
          std::make_unique<PartialIndexVector>(indexVector.getNumElements());
        std::unique_ptr<ILabelBinning<float64>> labelBinningPtr = labelBinningFactoryPtr_->create64Bit();
        return std::make_unique<DecomposableDynamicPartialBinnedRuleEvaluation<
          DenseDecomposableStatisticVector<float64>, CompleteIndexVector>>(
          indexVector, std::move(indexVectorPtr), threshold_, exponent_, l1RegularizationWeight_,
          l2RegularizationWeight_, std::move(labelBinningPtr));
    }

    std::unique_ptr<IRuleEvaluation<DenseDecomposableStatisticVector<float64>>>
      DecomposableDynamicPartialBinnedRuleEvaluationFactory::create(
        const DenseDecomposableStatisticVector<float64>& statisticVector, const PartialIndexVector& indexVector) const {
        std::unique_ptr<ILabelBinning<float64>> labelBinningPtr = labelBinningFactoryPtr_->create64Bit();
        return std::make_unique<
          DecomposableCompleteBinnedRuleEvaluation<DenseDecomposableStatisticVector<float64>, PartialIndexVector>>(
          indexVector, l1RegularizationWeight_, l2RegularizationWeight_, std::move(labelBinningPtr));
    }

    std::unique_ptr<IRuleEvaluation<SparseDecomposableStatisticVector<float32, uint32>>>
      DecomposableDynamicPartialBinnedRuleEvaluationFactory::create(
        const SparseDecomposableStatisticVector<float32, uint32>& statisticVector,
        const CompleteIndexVector& indexVector) const {
        std::unique_ptr<PartialIndexVector> indexVectorPtr =
          std::make_unique<PartialIndexVector>(indexVector.getNumElements());
        std::unique_ptr<ILabelBinning<float32>> labelBinningPtr = labelBinningFactoryPtr_->create32Bit();
        return std::make_unique<DecomposableDynamicPartialBinnedRuleEvaluation<
          SparseDecomposableStatisticVector<float32, uint32>, CompleteIndexVector>>(
          indexVector, std::move(indexVectorPtr), threshold_, exponent_, l1RegularizationWeight_,
          l2RegularizationWeight_, std::move(labelBinningPtr));
    }

    std::unique_ptr<IRuleEvaluation<SparseDecomposableStatisticVector<float32, uint32>>>
      DecomposableDynamicPartialBinnedRuleEvaluationFactory::create(
        const SparseDecomposableStatisticVector<float32, uint32>& statisticVector,
        const PartialIndexVector& indexVector) const {
        std::unique_ptr<ILabelBinning<float32>> labelBinningPtr = labelBinningFactoryPtr_->create32Bit();
        return std::make_unique<DecomposableCompleteBinnedRuleEvaluation<
          SparseDecomposableStatisticVector<float32, uint32>, PartialIndexVector>>(
          indexVector, l1RegularizationWeight_, l2RegularizationWeight_, std::move(labelBinningPtr));
    }

    std::unique_ptr<IRuleEvaluation<SparseDecomposableStatisticVector<float32, float32>>>
      DecomposableDynamicPartialBinnedRuleEvaluationFactory::create(
        const SparseDecomposableStatisticVector<float32, float32>& statisticVector,
        const CompleteIndexVector& indexVector) const {
        std::unique_ptr<PartialIndexVector> indexVectorPtr =
          std::make_unique<PartialIndexVector>(indexVector.getNumElements());
        std::unique_ptr<ILabelBinning<float32>> labelBinningPtr = labelBinningFactoryPtr_->create32Bit();
        return std::make_unique<DecomposableDynamicPartialBinnedRuleEvaluation<
          SparseDecomposableStatisticVector<float32, float32>, CompleteIndexVector>>(
          indexVector, std::move(indexVectorPtr), threshold_, exponent_, l1RegularizationWeight_,
          l2RegularizationWeight_, std::move(labelBinningPtr));
    }

    std::unique_ptr<IRuleEvaluation<SparseDecomposableStatisticVector<float32, float32>>>
      DecomposableDynamicPartialBinnedRuleEvaluationFactory::create(
        const SparseDecomposableStatisticVector<float32, float32>& statisticVector,
        const PartialIndexVector& indexVector) const {
        std::unique_ptr<ILabelBinning<float32>> labelBinningPtr = labelBinningFactoryPtr_->create32Bit();
        return std::make_unique<DecomposableCompleteBinnedRuleEvaluation<
          SparseDecomposableStatisticVector<float32, float32>, PartialIndexVector>>(
          indexVector, l1RegularizationWeight_, l2RegularizationWeight_, std::move(labelBinningPtr));
    }

    std::unique_ptr<IRuleEvaluation<SparseDecomposableStatisticVector<float64, uint32>>>
      DecomposableDynamicPartialBinnedRuleEvaluationFactory::create(
        const SparseDecomposableStatisticVector<float64, uint32>& statisticVector,
        const CompleteIndexVector& indexVector) const {
        std::unique_ptr<PartialIndexVector> indexVectorPtr =
          std::make_unique<PartialIndexVector>(indexVector.getNumElements());
        std::unique_ptr<ILabelBinning<float64>> labelBinningPtr = labelBinningFactoryPtr_->create64Bit();
        return std::make_unique<DecomposableDynamicPartialBinnedRuleEvaluation<
          SparseDecomposableStatisticVector<float64, uint32>, CompleteIndexVector>>(
          indexVector, std::move(indexVectorPtr), threshold_, exponent_, l1RegularizationWeight_,
          l2RegularizationWeight_, std::move(labelBinningPtr));
    }

    std::unique_ptr<IRuleEvaluation<SparseDecomposableStatisticVector<float64, uint32>>>
      DecomposableDynamicPartialBinnedRuleEvaluationFactory::create(
        const SparseDecomposableStatisticVector<float64, uint32>& statisticVector,
        const PartialIndexVector& indexVector) const {
        std::unique_ptr<ILabelBinning<float64>> labelBinningPtr = labelBinningFactoryPtr_->create64Bit();
        return std::make_unique<DecomposableCompleteBinnedRuleEvaluation<
          SparseDecomposableStatisticVector<float64, uint32>, PartialIndexVector>>(
          indexVector, l1RegularizationWeight_, l2RegularizationWeight_, std::move(labelBinningPtr));
    }

    std::unique_ptr<IRuleEvaluation<SparseDecomposableStatisticVector<float64, float32>>>
      DecomposableDynamicPartialBinnedRuleEvaluationFactory::create(
        const SparseDecomposableStatisticVector<float64, float32>& statisticVector,
        const CompleteIndexVector& indexVector) const {
        std::unique_ptr<PartialIndexVector> indexVectorPtr =
          std::make_unique<PartialIndexVector>(indexVector.getNumElements());
        std::unique_ptr<ILabelBinning<float64>> labelBinningPtr = labelBinningFactoryPtr_->create64Bit();
        return std::make_unique<DecomposableDynamicPartialBinnedRuleEvaluation<
          SparseDecomposableStatisticVector<float64, float32>, CompleteIndexVector>>(
          indexVector, std::move(indexVectorPtr), threshold_, exponent_, l1RegularizationWeight_,
          l2RegularizationWeight_, std::move(labelBinningPtr));
    }

    std::unique_ptr<IRuleEvaluation<SparseDecomposableStatisticVector<float64, float32>>>
      DecomposableDynamicPartialBinnedRuleEvaluationFactory::create(
        const SparseDecomposableStatisticVector<float64, float32>& statisticVector,
        const PartialIndexVector& indexVector) const {
        std::unique_ptr<ILabelBinning<float64>> labelBinningPtr = labelBinningFactoryPtr_->create64Bit();
        return std::make_unique<DecomposableCompleteBinnedRuleEvaluation<
          SparseDecomposableStatisticVector<float64, float32>, PartialIndexVector>>(
          indexVector, l1RegularizationWeight_, l2RegularizationWeight_, std::move(labelBinningPtr));
    }
}
