#include "boosting/rule_evaluation/rule_evaluation_label_wise_partial_fixed_binned.hpp"
#include "rule_evaluation_label_wise_partial_fixed_common.hpp"
#include "rule_evaluation_label_wise_binned_common.hpp"


namespace boosting {

    /**
     * Allows to calculate the predictions of partial rules that predict for a predefined number of labels, as well as
     * an overall quality score, based on the gradients and Hessians that are stored by a
     * `DenseLabelWiseStatisticVector` using L1 and L2 regularization. The labels are assigned to bins based on the
     * gradients and Hessians.
     *
     * @tparam IndexVector The type of the vector that provides access to the labels for which predictions should be
     *                     calculated
     */
    template<typename IndexVector>
    class DenseLabelWiseFixedPartialBinnedRuleEvaluation final :
            public AbstractLabelWiseBinnedRuleEvaluation<DenseLabelWiseStatisticVector, PartialIndexVector> {

        private:

            const IndexVector& labelIndices_;

            std::unique_ptr<PartialIndexVector> indexVectorPtr_;

            SparseArrayVector<float64> tmpVector_;

        protected:

            uint32 calculateLabelWiseCriteria(const DenseLabelWiseStatisticVector& statisticVector, float64* criteria,
                                            uint32 numCriteria, float64 l1RegularizationWeight,
                                            float64 l2RegularizationWeight) override {
                uint32 numElements = statisticVector.getNumElements();
                DenseLabelWiseStatisticVector::const_iterator statisticIterator = statisticVector.cbegin();
                SparseArrayVector<float64>::iterator tmpIterator = tmpVector_.begin();
                sortLabelWiseScores(tmpIterator, statisticIterator, numElements, numCriteria, l1RegularizationWeight,
                                    l2RegularizationWeight);
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
             * @param indexVectorPtr            An unique pointer to an object of type `PartialIndexVector` that stores
             *                                  the indices of the labels for which a rule predicts
             * @param l1RegularizationWeight    The weight of the L1 regularization that is applied for calculating the
             *                                  scores to be predicted by rules
             * @param l2RegularizationWeight    The weight of the L2 regularization that is applied for calculating the
             *                                  scores to be predicted by rules
             * @param binningPtr                An unique pointer to an object of type `ILabelBinning` that should be
             *                                  used to assign labels to bins
             */
            DenseLabelWiseFixedPartialBinnedRuleEvaluation(const IndexVector& labelIndices,
                                                           std::unique_ptr<PartialIndexVector> indexVectorPtr,
                                                           float64 l1RegularizationWeight,
                                                           float64 l2RegularizationWeight,
                                                           std::unique_ptr<ILabelBinning> binningPtr)
                : AbstractLabelWiseBinnedRuleEvaluation<DenseLabelWiseStatisticVector, PartialIndexVector>(
                      *indexVectorPtr, false, l1RegularizationWeight, l2RegularizationWeight, std::move(binningPtr)),
                  labelIndices_(labelIndices), indexVectorPtr_(std::move(indexVectorPtr)),
                  tmpVector_(SparseArrayVector<float64>(labelIndices.getNumElements())) {

            }

    };

    LabelWiseFixedPartialBinnedRuleEvaluationFactory::LabelWiseFixedPartialBinnedRuleEvaluationFactory(
            float32 labelRatio, uint32 minLabels, uint32 maxLabels, float64 l1RegularizationWeight,
            float64 l2RegularizationWeight, std::unique_ptr<ILabelBinningFactory> labelBinningFactoryPtr)
        : labelRatio_(labelRatio), minLabels_(minLabels), maxLabels_(maxLabels),
          l1RegularizationWeight_(l1RegularizationWeight), l2RegularizationWeight_(l2RegularizationWeight),
          labelBinningFactoryPtr_(std::move(labelBinningFactoryPtr)) {

    }

    std::unique_ptr<IRuleEvaluation<DenseLabelWiseStatisticVector>> LabelWiseFixedPartialBinnedRuleEvaluationFactory::create(
            const DenseLabelWiseStatisticVector& statisticVector, const CompleteIndexVector& indexVector) const {
        std::unique_ptr<PartialIndexVector> indexVectorPtr = std::make_unique<PartialIndexVector>(
            calculateBoundedFraction(indexVector.getNumElements(), labelRatio_, minLabels_, maxLabels_));
        std::unique_ptr<ILabelBinning> labelBinningPtr = labelBinningFactoryPtr_->create();
        return std::make_unique<DenseLabelWiseFixedPartialBinnedRuleEvaluation<CompleteIndexVector>>(
            indexVector, std::move(indexVectorPtr), l1RegularizationWeight_, l2RegularizationWeight_,
            std::move(labelBinningPtr));
    }

    std::unique_ptr<IRuleEvaluation<DenseLabelWiseStatisticVector>> LabelWiseFixedPartialBinnedRuleEvaluationFactory::create(
            const DenseLabelWiseStatisticVector& statisticVector, const PartialIndexVector& indexVector) const {
        std::unique_ptr<ILabelBinning> labelBinningPtr = labelBinningFactoryPtr_->create();
        return std::make_unique<DenseLabelWiseCompleteBinnedRuleEvaluation<PartialIndexVector>>(
            indexVector, l1RegularizationWeight_, l2RegularizationWeight_, std::move(labelBinningPtr));
    }

}
