#include "boosting/rule_evaluation/rule_evaluation_label_wise_single.hpp"
#include "common/rule_evaluation/score_vector_dense.hpp"


namespace boosting {

    /**
     * Allows to calculate the predictions of rules, as well as an overall quality score, based on the gradients and
     * Hessians that are stored by a `DenseLabelWiseStatisticVector` using L2 regularization.
     *
     * @tparam T The type of the vector that provides access to the labels for which predictions should be calculated
     */
    template<typename T>
    class LabelWiseSingleLabelRuleEvaluation final : public ILabelWiseRuleEvaluation<DenseLabelWiseStatisticVector> {

        private:

            DenseScoreVector<T> scoreVector_;

            float64 l2RegularizationWeight_;

        public:

            /**
             * @param labelIndices              A reference to an object of template type `T` that provides access to
             *                                  the indices of the labels for which the rules may predict
             * @param l2RegularizationWeight    The weight of the L2 regularization that is applied for calculating the
             *                                  scores to be predicted by rules
             */
            LabelWiseSingleLabelRuleEvaluation(const T& labelIndices, float64 l2RegularizationWeight)
                : scoreVector_(DenseScoreVector<T>(labelIndices)), l2RegularizationWeight_(l2RegularizationWeight) {

            }

            const ILabelWiseScoreVector& calculateLabelWisePrediction(
                    const DenseLabelWiseStatisticVector& statisticVector) override {

            }

            const IScoreVector& calculatePrediction(const DenseLabelWiseStatisticVector& statisticVector) override {
                // TODO Implement
                return scoreVector_;
            }

    };

    LabelWiseSingleLabelRuleEvaluationFactory::LabelWiseSingleLabelRuleEvaluationFactory(float64 l2RegularizationWeight)
        : l2RegularizationWeight_(l2RegularizationWeight) {

    }

    std::unique_ptr<ILabelWiseRuleEvaluation<DenseLabelWiseStatisticVector>> LabelWiseSingleLabelRuleEvaluationFactory::createDense(
            const CompleteIndexVector& indexVector) const {
        return std::make_unique<LabelWiseSingleLabelRuleEvaluation<CompleteIndexVector>>(indexVector,
                                                                                         l2RegularizationWeight_);
    }

    std::unique_ptr<ILabelWiseRuleEvaluation<DenseLabelWiseStatisticVector>> LabelWiseSingleLabelRuleEvaluationFactory::createDense(
            const PartialIndexVector& indexVector) const {
        return std::make_unique<LabelWiseSingleLabelRuleEvaluation<PartialIndexVector>>(indexVector,
                                                                                        l2RegularizationWeight_);
    }

}