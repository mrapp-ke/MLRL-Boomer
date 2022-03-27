#include "boosting/rule_evaluation/rule_evaluation_label_wise_partial_dynamic_binned.hpp"
#include "rule_evaluation_label_wise_binned_common.hpp"


namespace boosting {

    /**
     * Allows to calculate the predictions of partial rules that predict for a subset of the available labels that is
     * determined dynamically, as well as an overall quality score, based on the gradients and Hessians that are stored
     * by a `DenseLabelWiseStatisticVector` using L1 and L2 regularization. The labels are assigned to bins based on the
     * gradients and Hessians.
     *
     * @tparam T The type of the vector that provides access to the labels for which predictions should be calculated
     */
    template<typename T>
    class DenseLabelWiseDynamicPartialBinnedRuleEvaluation final :
            public AbstractLabelWiseBinnedRuleEvaluation<DenseLabelWiseStatisticVector, PartialIndexVector> {

        private:

            const T& labelIndices_;

            std::unique_ptr<PartialIndexVector> indexVectorPtr_;

            float64 threshold_;

        protected:

            uint32 calculateLabelWiseCriteria(const DenseLabelWiseStatisticVector& statisticVector, float64* criteria,
                                              uint32 numCriteria, float64 l1RegularizationWeight,
                                              float64 l2RegularizationWeight) override {
                uint32 numElements = statisticVector.getNumElements();
                DenseLabelWiseStatisticVector::const_iterator statisticIterator = statisticVector.cbegin();
                const Tuple<float64>& firstTuple = statisticIterator[0];
                float64 bestQualityScore = calculateLabelWiseQualityScore(firstTuple.first, firstTuple.second,
                                                                          l1RegularizationWeight,
                                                                          l2RegularizationWeight);

                for (uint32 i = 1; i < numElements; i++) {
                    const Tuple<float64>& tuple = statisticIterator[i];
                    float64 qualityScore = calculateLabelWiseQualityScore(tuple.first, tuple.second,
                                                                          l1RegularizationWeight,
                                                                          l2RegularizationWeight);

                    if (qualityScore < bestQualityScore) {
                        bestQualityScore = qualityScore;
                    }
                }

                float64 threshold = (bestQualityScore * bestQualityScore) * threshold_;
                typename T::const_iterator labelIndexIterator = labelIndices_.cbegin();
                PartialIndexVector::iterator indexIterator = indexVectorPtr_->begin();
                uint32 n = 0;

                for (uint32 i = 0; i < numElements; i++) {
                    const Tuple<float64>& tuple = statisticIterator[i];
                    float64 qualityScore = calculateLabelWiseQualityScore(tuple.first, tuple.second,
                                                                          l1RegularizationWeight,
                                                                          l2RegularizationWeight);

                    if (qualityScore * qualityScore > threshold) {
                        criteria[n] = calculateLabelWiseScore(tuple.first, tuple.second, l1RegularizationWeight,
                                                              l2RegularizationWeight);
                        indexIterator[n] = labelIndexIterator[i];
                        n++;
                    }
                }

                indexVectorPtr_->setNumElements(n, false);
                return n;
            }

        public:

            /**
             * @param labelIndices              A reference to an object of template type `T` that provides access to
             *                                  the indices of the labels for which the rules may predict
             * @param indexVectorPtr            An unique pointer to an object of type `PartialIndexVector` that stores
             *                                  the indices of the labels for which a rule predicts
             * @param threshold                 A threshold that affects for how many labels the rule heads should
             *                                  predict
             * @param l1RegularizationWeight    The weight of the L1 regularization that is applied for calculating the
             *                                  scores to be predicted by rules
             * @param l2RegularizationWeight    The weight of the L2 regularization that is applied for calculating the
             *                                  scores to be predicted by rules
             * @param binningPtr                An unique pointer to an object of type `ILabelBinning` that should be
             *                                  used to assign labels to bins
             */
            DenseLabelWiseDynamicPartialBinnedRuleEvaluation(const T& labelIndices,
                                                             std::unique_ptr<PartialIndexVector> indexVectorPtr,
                                                             float32 threshold, float64 l1RegularizationWeight,
                                                             float64 l2RegularizationWeight,
                                                             std::unique_ptr<ILabelBinning> binningPtr)
                : AbstractLabelWiseBinnedRuleEvaluation<DenseLabelWiseStatisticVector, PartialIndexVector>(
                      *indexVectorPtr, true, l1RegularizationWeight, l2RegularizationWeight, std::move(binningPtr)),
                  labelIndices_(labelIndices), indexVectorPtr_(std::move(indexVectorPtr)), threshold_(1.0 - threshold) {

            }

    };

    LabelWiseDynamicPartialBinnedRuleEvaluationFactory::LabelWiseDynamicPartialBinnedRuleEvaluationFactory(
            float32 threshold, float64 l1RegularizationWeight, float64 l2RegularizationWeight,
            std::unique_ptr<ILabelBinningFactory> labelBinningFactoryPtr)
        : threshold_(threshold), l1RegularizationWeight_(l1RegularizationWeight),
          l2RegularizationWeight_(l2RegularizationWeight), labelBinningFactoryPtr_(std::move(labelBinningFactoryPtr)) {

    }

    std::unique_ptr<IRuleEvaluation<DenseLabelWiseStatisticVector>> LabelWiseDynamicPartialBinnedRuleEvaluationFactory::create(
            const DenseLabelWiseStatisticVector& statisticVector, const CompleteIndexVector& indexVector) const {
        std::unique_ptr<PartialIndexVector> indexVectorPtr =
            std::make_unique<PartialIndexVector>(indexVector.getNumElements());
        std::unique_ptr<ILabelBinning> labelBinningPtr = labelBinningFactoryPtr_->create();
        return std::make_unique<DenseLabelWiseDynamicPartialBinnedRuleEvaluation<CompleteIndexVector>>(
            indexVector, std::move(indexVectorPtr), threshold_, l1RegularizationWeight_, l2RegularizationWeight_,
            std::move(labelBinningPtr));
    }

    std::unique_ptr<IRuleEvaluation<DenseLabelWiseStatisticVector>> LabelWiseDynamicPartialBinnedRuleEvaluationFactory::create(
            const DenseLabelWiseStatisticVector& statisticVector, const PartialIndexVector& indexVector) const {
        std::unique_ptr<ILabelBinning> labelBinningPtr = labelBinningFactoryPtr_->create();
        return std::make_unique<DenseLabelWiseCompleteBinnedRuleEvaluation<PartialIndexVector>>(
            indexVector, l1RegularizationWeight_, l2RegularizationWeight_, std::move(labelBinningPtr));
    }

}
