#include "mlrl/boosting/rule_evaluation/rule_evaluation_decomposable_partial_dynamic_binned.hpp"

#include "mlrl/boosting/rule_evaluation/simd/vector_math_decomposable_simd.hpp"
#include "mlrl/boosting/rule_evaluation/vector_math_decomposable.hpp"
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
     * @tparam VectorMath       The type that implements basic operations for calculating with gradients and Hessians
     */
    template<typename StatisticVector, typename IndexVector, typename VectorMath>
    class DecomposableDynamicPartialBinnedRuleEvaluation final
        : public AbstractDecomposableBinnedRuleEvaluation<StatisticVector, PartialIndexVector> {
        private:

            using statistic_type = StatisticVector::statistic_type;

            const IndexVector& labelIndices_;

            const std::unique_ptr<PartialIndexVector> indexVectorPtr_;

            const float32 threshold_;

            const float32 exponent_;

            template<typename StatisticType, typename WeightType>
            static inline void calculateOutputWiseCriteriaInternally(
              const IndexVector& labelIndices,
              const SparseDecomposableStatisticVectorView<StatisticType, WeightType>& statisticVector,
              PartialIndexVector& indexVector, typename View<statistic_type>::iterator criteria,
              float32 l1RegularizationWeight, float32 l2RegularizationWeight, float32 threshold, float32 exponent) {
                uint32 numElements = statisticVector.getNumElements();
                auto gradientIterator = statisticVector.gradients_cbegin();
                auto hessianIterator = statisticVector.hessians_cbegin();
                const std::pair<statistic_type, statistic_type> pair = getMinAndMaxScore(
                  gradientIterator, hessianIterator, numElements, l1RegularizationWeight, l2RegularizationWeight);
                statistic_type minAbsScore = pair.first;
                statistic_type scoreThreshold = calculateThreshold(minAbsScore, pair.second, threshold, exponent);
                PartialIndexVector::iterator indexIterator = indexVector.begin();
                typename IndexVector::const_iterator labelIndexIterator = labelIndices.cbegin();
                uint32 n = 0;

                for (uint32 i = 0; i < numElements; i++) {
                    statistic_type score = calculateOutputWiseScore(gradientIterator[i], hessianIterator[i],
                                                                    l1RegularizationWeight, l2RegularizationWeight);

                    if (calculateWeightedScore(score, minAbsScore, exponent) >= scoreThreshold) {
                        indexIterator[n] = labelIndexIterator[i];
                        criteria[n] = score;
                        n++;
                    }
                }

                indexVector.setNumElements(n, false);
            }

            template<typename StatisticType>
            static inline void calculateOutputWiseCriteriaInternally(
              const IndexVector& labelIndices,
              const DenseDecomposableStatisticVectorView<StatisticType>& statisticVector,
              PartialIndexVector& indexVector, typename View<statistic_type>::iterator criteria,
              float32 l1RegularizationWeight, float32 l2RegularizationWeight, float32 threshold, float32 exponent) {
                uint32 numElements = statisticVector.getNumElements();
                auto gradientIterator = statisticVector.gradients_cbegin();
                auto hessianIterator = statisticVector.hessians_cbegin();

                VectorMath::calculateOutputWiseScores(gradientIterator, hessianIterator, criteria, numElements,
                                                      l1RegularizationWeight, l2RegularizationWeight);

                const std::pair<statistic_type, statistic_type> pair = getMinAndMaxScore(
                  gradientIterator, hessianIterator, numElements, l1RegularizationWeight, l2RegularizationWeight);
                statistic_type minAbsScore = pair.first;
                statistic_type scoreThreshold = calculateThreshold(minAbsScore, pair.second, threshold, exponent);
                PartialIndexVector::iterator indexIterator = indexVector.begin();
                typename IndexVector::const_iterator labelIndexIterator = labelIndices.cbegin();
                uint32 n = 0;

                for (uint32 i = 0; i < numElements; i++) {
                    statistic_type score = criteria[i];

                    if (calculateWeightedScore(score, minAbsScore, exponent) >= scoreThreshold) {
                        indexIterator[n] = labelIndexIterator[i];
                        criteria[n] = score;
                        n++;
                    }
                }

                indexVector.setNumElements(n, false);
            }

        protected:

            uint32 calculateOutputWiseCriteria(const StatisticVector& statisticVector,
                                               typename View<statistic_type>::iterator criteria, uint32 numCriteria,
                                               float32 l1RegularizationWeight,
                                               float32 l2RegularizationWeight) override {
                calculateOutputWiseCriteriaInternally(labelIndices_, statisticVector, *indexVectorPtr_, criteria,
                                                      l1RegularizationWeight, l2RegularizationWeight, threshold_,
                                                      exponent_);
                return indexVectorPtr_->getNumElements();
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

    template<typename VectorMath>
    DecomposableDynamicPartialBinnedRuleEvaluationFactory<VectorMath>::
      DecomposableDynamicPartialBinnedRuleEvaluationFactory(
        float32 threshold, float32 exponent, float32 l1RegularizationWeight, float32 l2RegularizationWeight,
        std::unique_ptr<ILabelBinningFactory> labelBinningFactoryPtr)
        : threshold_(threshold), exponent_(exponent), l1RegularizationWeight_(l1RegularizationWeight),
          l2RegularizationWeight_(l2RegularizationWeight), labelBinningFactoryPtr_(std::move(labelBinningFactoryPtr)) {}

    template<typename VectorMath>
    std::unique_ptr<IRuleEvaluation<DenseDecomposableStatisticVectorView<float32>>>
      DecomposableDynamicPartialBinnedRuleEvaluationFactory<VectorMath>::create(
        const DenseDecomposableStatisticVectorView<float32>& statisticVector,
        const CompleteIndexVector& indexVector) const {
        std::unique_ptr<PartialIndexVector> indexVectorPtr =
          std::make_unique<PartialIndexVector>(indexVector.getNumElements());
        std::unique_ptr<ILabelBinning<float32>> labelBinningPtr = labelBinningFactoryPtr_->create32Bit();
        return std::make_unique<DecomposableDynamicPartialBinnedRuleEvaluation<
          DenseDecomposableStatisticVectorView<float32>, CompleteIndexVector, VectorMath>>(
          indexVector, std::move(indexVectorPtr), threshold_, exponent_, l1RegularizationWeight_,
          l2RegularizationWeight_, std::move(labelBinningPtr));
    }

    template<typename VectorMath>
    std::unique_ptr<IRuleEvaluation<DenseDecomposableStatisticVectorView<float32>>>
      DecomposableDynamicPartialBinnedRuleEvaluationFactory<VectorMath>::create(
        const DenseDecomposableStatisticVectorView<float32>& statisticVector,
        const PartialIndexVector& indexVector) const {
        std::unique_ptr<ILabelBinning<float32>> labelBinningPtr = labelBinningFactoryPtr_->create32Bit();
        return std::make_unique<
          DecomposableCompleteBinnedRuleEvaluation<DenseDecomposableStatisticVectorView<float32>, PartialIndexVector>>(
          indexVector, l1RegularizationWeight_, l2RegularizationWeight_, std::move(labelBinningPtr));
    }

    template<typename VectorMath>
    std::unique_ptr<IRuleEvaluation<DenseDecomposableStatisticVectorView<float64>>>
      DecomposableDynamicPartialBinnedRuleEvaluationFactory<VectorMath>::create(
        const DenseDecomposableStatisticVectorView<float64>& statisticVector,
        const CompleteIndexVector& indexVector) const {
        std::unique_ptr<PartialIndexVector> indexVectorPtr =
          std::make_unique<PartialIndexVector>(indexVector.getNumElements());
        std::unique_ptr<ILabelBinning<float64>> labelBinningPtr = labelBinningFactoryPtr_->create64Bit();
        return std::make_unique<DecomposableDynamicPartialBinnedRuleEvaluation<
          DenseDecomposableStatisticVectorView<float64>, CompleteIndexVector, VectorMath>>(
          indexVector, std::move(indexVectorPtr), threshold_, exponent_, l1RegularizationWeight_,
          l2RegularizationWeight_, std::move(labelBinningPtr));
    }

    template<typename VectorMath>
    std::unique_ptr<IRuleEvaluation<DenseDecomposableStatisticVectorView<float64>>>
      DecomposableDynamicPartialBinnedRuleEvaluationFactory<VectorMath>::create(
        const DenseDecomposableStatisticVectorView<float64>& statisticVector,
        const PartialIndexVector& indexVector) const {
        std::unique_ptr<ILabelBinning<float64>> labelBinningPtr = labelBinningFactoryPtr_->create64Bit();
        return std::make_unique<
          DecomposableCompleteBinnedRuleEvaluation<DenseDecomposableStatisticVectorView<float64>, PartialIndexVector>>(
          indexVector, l1RegularizationWeight_, l2RegularizationWeight_, std::move(labelBinningPtr));
    }

    template<typename VectorMath>
    std::unique_ptr<IRuleEvaluation<SparseDecomposableStatisticVectorView<float32, uint32>>>
      DecomposableDynamicPartialBinnedRuleEvaluationFactory<VectorMath>::create(
        const SparseDecomposableStatisticVectorView<float32, uint32>& statisticVector,
        const CompleteIndexVector& indexVector) const {
        std::unique_ptr<PartialIndexVector> indexVectorPtr =
          std::make_unique<PartialIndexVector>(indexVector.getNumElements());
        std::unique_ptr<ILabelBinning<float32>> labelBinningPtr = labelBinningFactoryPtr_->create32Bit();
        return std::make_unique<DecomposableDynamicPartialBinnedRuleEvaluation<
          SparseDecomposableStatisticVectorView<float32, uint32>, CompleteIndexVector, VectorMath>>(
          indexVector, std::move(indexVectorPtr), threshold_, exponent_, l1RegularizationWeight_,
          l2RegularizationWeight_, std::move(labelBinningPtr));
    }

    template<typename VectorMath>
    std::unique_ptr<IRuleEvaluation<SparseDecomposableStatisticVectorView<float32, uint32>>>
      DecomposableDynamicPartialBinnedRuleEvaluationFactory<VectorMath>::create(
        const SparseDecomposableStatisticVectorView<float32, uint32>& statisticVector,
        const PartialIndexVector& indexVector) const {
        std::unique_ptr<ILabelBinning<float32>> labelBinningPtr = labelBinningFactoryPtr_->create32Bit();
        return std::make_unique<DecomposableCompleteBinnedRuleEvaluation<
          SparseDecomposableStatisticVectorView<float32, uint32>, PartialIndexVector>>(
          indexVector, l1RegularizationWeight_, l2RegularizationWeight_, std::move(labelBinningPtr));
    }

    template<typename VectorMath>
    std::unique_ptr<IRuleEvaluation<SparseDecomposableStatisticVectorView<float32, float32>>>
      DecomposableDynamicPartialBinnedRuleEvaluationFactory<VectorMath>::create(
        const SparseDecomposableStatisticVectorView<float32, float32>& statisticVector,
        const CompleteIndexVector& indexVector) const {
        std::unique_ptr<PartialIndexVector> indexVectorPtr =
          std::make_unique<PartialIndexVector>(indexVector.getNumElements());
        std::unique_ptr<ILabelBinning<float32>> labelBinningPtr = labelBinningFactoryPtr_->create32Bit();
        return std::make_unique<DecomposableDynamicPartialBinnedRuleEvaluation<
          SparseDecomposableStatisticVectorView<float32, float32>, CompleteIndexVector, VectorMath>>(
          indexVector, std::move(indexVectorPtr), threshold_, exponent_, l1RegularizationWeight_,
          l2RegularizationWeight_, std::move(labelBinningPtr));
    }

    template<typename VectorMath>
    std::unique_ptr<IRuleEvaluation<SparseDecomposableStatisticVectorView<float32, float32>>>
      DecomposableDynamicPartialBinnedRuleEvaluationFactory<VectorMath>::create(
        const SparseDecomposableStatisticVectorView<float32, float32>& statisticVector,
        const PartialIndexVector& indexVector) const {
        std::unique_ptr<ILabelBinning<float32>> labelBinningPtr = labelBinningFactoryPtr_->create32Bit();
        return std::make_unique<DecomposableCompleteBinnedRuleEvaluation<
          SparseDecomposableStatisticVectorView<float32, float32>, PartialIndexVector>>(
          indexVector, l1RegularizationWeight_, l2RegularizationWeight_, std::move(labelBinningPtr));
    }

    template<typename VectorMath>
    std::unique_ptr<IRuleEvaluation<SparseDecomposableStatisticVectorView<float64, uint32>>>
      DecomposableDynamicPartialBinnedRuleEvaluationFactory<VectorMath>::create(
        const SparseDecomposableStatisticVectorView<float64, uint32>& statisticVector,
        const CompleteIndexVector& indexVector) const {
        std::unique_ptr<PartialIndexVector> indexVectorPtr =
          std::make_unique<PartialIndexVector>(indexVector.getNumElements());
        std::unique_ptr<ILabelBinning<float64>> labelBinningPtr = labelBinningFactoryPtr_->create64Bit();
        return std::make_unique<DecomposableDynamicPartialBinnedRuleEvaluation<
          SparseDecomposableStatisticVectorView<float64, uint32>, CompleteIndexVector, VectorMath>>(
          indexVector, std::move(indexVectorPtr), threshold_, exponent_, l1RegularizationWeight_,
          l2RegularizationWeight_, std::move(labelBinningPtr));
    }

    template<typename VectorMath>
    std::unique_ptr<IRuleEvaluation<SparseDecomposableStatisticVectorView<float64, uint32>>>
      DecomposableDynamicPartialBinnedRuleEvaluationFactory<VectorMath>::create(
        const SparseDecomposableStatisticVectorView<float64, uint32>& statisticVector,
        const PartialIndexVector& indexVector) const {
        std::unique_ptr<ILabelBinning<float64>> labelBinningPtr = labelBinningFactoryPtr_->create64Bit();
        return std::make_unique<DecomposableCompleteBinnedRuleEvaluation<
          SparseDecomposableStatisticVectorView<float64, uint32>, PartialIndexVector>>(
          indexVector, l1RegularizationWeight_, l2RegularizationWeight_, std::move(labelBinningPtr));
    }

    template<typename VectorMath>
    std::unique_ptr<IRuleEvaluation<SparseDecomposableStatisticVectorView<float64, float32>>>
      DecomposableDynamicPartialBinnedRuleEvaluationFactory<VectorMath>::create(
        const SparseDecomposableStatisticVectorView<float64, float32>& statisticVector,
        const CompleteIndexVector& indexVector) const {
        std::unique_ptr<PartialIndexVector> indexVectorPtr =
          std::make_unique<PartialIndexVector>(indexVector.getNumElements());
        std::unique_ptr<ILabelBinning<float64>> labelBinningPtr = labelBinningFactoryPtr_->create64Bit();
        return std::make_unique<DecomposableDynamicPartialBinnedRuleEvaluation<
          SparseDecomposableStatisticVectorView<float64, float32>, CompleteIndexVector, VectorMath>>(
          indexVector, std::move(indexVectorPtr), threshold_, exponent_, l1RegularizationWeight_,
          l2RegularizationWeight_, std::move(labelBinningPtr));
    }

    template<typename VectorMath>
    std::unique_ptr<IRuleEvaluation<SparseDecomposableStatisticVectorView<float64, float32>>>
      DecomposableDynamicPartialBinnedRuleEvaluationFactory<VectorMath>::create(
        const SparseDecomposableStatisticVectorView<float64, float32>& statisticVector,
        const PartialIndexVector& indexVector) const {
        std::unique_ptr<ILabelBinning<float64>> labelBinningPtr = labelBinningFactoryPtr_->create64Bit();
        return std::make_unique<DecomposableCompleteBinnedRuleEvaluation<
          SparseDecomposableStatisticVectorView<float64, float32>, PartialIndexVector>>(
          indexVector, l1RegularizationWeight_, l2RegularizationWeight_, std::move(labelBinningPtr));
    }

    template class DecomposableDynamicPartialBinnedRuleEvaluationFactory<SequentialDecomposableVectorMath>;
#if SIMD_SUPPORT_ENABLED
    template class DecomposableDynamicPartialBinnedRuleEvaluationFactory<SimdDecomposableVectorMath>;
#endif
}
