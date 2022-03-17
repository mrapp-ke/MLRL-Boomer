#include "boosting/rule_evaluation/rule_evaluation_label_wise_complete_binned.hpp"
#include "common/rule_evaluation/score_vector_binned_dense.hpp"
#include "common/data/arrays.hpp"
#include "rule_evaluation_label_wise_binned_common.hpp"


namespace boosting {

    /**
     * Allows to calculate the predictions of complete rules, as well as an overall quality score, based on the
     * gradients and Hessians that are stored by a `DenseLabelWiseStatisticVector` using L1 and L2 regularization. The
     * labels are assigned to bins based on the gradients and Hessians.
     *
     * @tparam T The type of the vector that provides access to the labels for which predictions should be calculated
     */
    template<typename T>
    class DenseLabelWiseCompleteBinnedRuleEvaluation final :
            public AbstractLabelWiseBinnedRuleEvaluation<DenseLabelWiseStatisticVector, T> {

        protected:

            void calculateLabelWiseCriteria(const DenseLabelWiseStatisticVector& statisticVector, float64* criteria,
                                            uint32 numCriteria, float64 l1RegularizationWeight,
                                            float64 l2RegularizationWeight) override {
                DenseLabelWiseStatisticVector::const_iterator statisticIterator = statisticVector.cbegin();

                for (uint32 i = 0; i < numCriteria; i++) {
                    const Tuple<float64>& tuple = statisticIterator[i];
                    criteria[i] = calculateLabelWiseScore(tuple.first, tuple.second, l1RegularizationWeight,
                                                          l2RegularizationWeight);
                }
            }

        public:

            /**
             * @param labelIndices              A reference to an object of template type `T` that provides access to
             *                                  the indices of the labels for which the rules may predict
             * @param l1RegularizationWeight    The weight of the L1 regularization that is applied for calculating the
             *                                  scores to be predicted by rules
             * @param l2RegularizationWeight    The weight of the L2 regularization that is applied for calculating the
             *                                  scores to be predicted by rules
             * @param binningPtr                An unique pointer to an object of type `ILabelBinning` that should be
             *                                  used to assign labels to bins
             */
            DenseLabelWiseCompleteBinnedRuleEvaluation(const T& labelIndices, float64 l1RegularizationWeight,
                                                       float64 l2RegularizationWeight,
                                                       std::unique_ptr<ILabelBinning> binningPtr)
                : AbstractLabelWiseBinnedRuleEvaluation<DenseLabelWiseStatisticVector, T>(labelIndices,
                                                                                          l1RegularizationWeight,
                                                                                          l2RegularizationWeight,
                                                                                          std::move(binningPtr)) {

            }

    };

    LabelWiseCompleteBinnedRuleEvaluationFactory::LabelWiseCompleteBinnedRuleEvaluationFactory(
            float64 l1RegularizationWeight, float64 l2RegularizationWeight,
            std::unique_ptr<ILabelBinningFactory> labelBinningFactoryPtr)
        : l1RegularizationWeight_(l1RegularizationWeight), l2RegularizationWeight_(l2RegularizationWeight),
          labelBinningFactoryPtr_(std::move(labelBinningFactoryPtr)) {

    }

    std::unique_ptr<IRuleEvaluation<DenseLabelWiseStatisticVector>> LabelWiseCompleteBinnedRuleEvaluationFactory::create(
            const DenseLabelWiseStatisticVector& statisticVector, const CompleteIndexVector& indexVector) const {
        std::unique_ptr<ILabelBinning> labelBinningPtr = labelBinningFactoryPtr_->create();
        return std::make_unique<DenseLabelWiseCompleteBinnedRuleEvaluation<CompleteIndexVector>>(
            indexVector, l1RegularizationWeight_, l2RegularizationWeight_, std::move(labelBinningPtr));
    }

    std::unique_ptr<IRuleEvaluation<DenseLabelWiseStatisticVector>> LabelWiseCompleteBinnedRuleEvaluationFactory::create(
            const DenseLabelWiseStatisticVector& statisticVector, const PartialIndexVector& indexVector) const {
        std::unique_ptr<ILabelBinning> labelBinningPtr = labelBinningFactoryPtr_->create();
        return std::make_unique<DenseLabelWiseCompleteBinnedRuleEvaluation<PartialIndexVector>>(
            indexVector, l1RegularizationWeight_, l2RegularizationWeight_, std::move(labelBinningPtr));
    }

}