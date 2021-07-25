#include "boosting/rule_evaluation/rule_evaluation_label_wise_regularized.hpp"
#include "common/rule_evaluation/score_vector_label_wise_dense.hpp"
#include "common/validation.hpp"
#include "rule_evaluation_label_wise_regularized_common.hpp"


namespace boosting {

    /**
     * Allows to calculate the predictions of rules, as well as corresponding quality scores, based on the gradients and
     * Hessians that are stored by a `DenseLabelWiseStatisticVector` using L2 regularization.
     *
     * @tparam T The type of the vector that provides access to the labels for which predictions should be calculated
     */
    template<typename T>
    class DenseRegularizedLabelWiseRuleEvaluation final : public ILabelWiseRuleEvaluation<DenseLabelWiseStatisticVector> {

        private:

            float64 l2RegularizationWeight_;

            DenseLabelWiseScoreVector<T> scoreVector_;

        public:

            /**
             * @param labelIndices              A reference to an object of template type `T` that provides access to
             *                                  the indices of the labels for which the rules may predict
             * @param l2RegularizationWeight    The weight of the L2 regularization that is applied for calculating the
             *                                  scores to be predicted by rules
             */
            DenseRegularizedLabelWiseRuleEvaluation(const T& labelIndices, float64 l2RegularizationWeight)
                : l2RegularizationWeight_(l2RegularizationWeight),
                  scoreVector_(DenseLabelWiseScoreVector<T>(labelIndices)) {

            }

            const ILabelWiseScoreVector& calculateLabelWisePrediction(
                    const DenseLabelWiseStatisticVector& statisticVector) override {
                scoreVector_.overallQualityScore = calculateLabelWisePredictionInternally<
                        typename DenseLabelWiseScoreVector<T>::score_iterator,
                        typename DenseLabelWiseScoreVector<T>::quality_score_iterator,
                        DenseLabelWiseStatisticVector::gradient_const_iterator,
                        DenseLabelWiseStatisticVector::hessian_const_iterator>(
                    scoreVector_.getNumElements(), scoreVector_.scores_begin(), scoreVector_.quality_scores_begin(),
                    statisticVector.gradients_cbegin(), statisticVector.hessians_cbegin(), l2RegularizationWeight_);
                return scoreVector_;
            }

    };

    RegularizedLabelWiseRuleEvaluationFactory::RegularizedLabelWiseRuleEvaluationFactory(float64 l2RegularizationWeight)
        : l2RegularizationWeight_(l2RegularizationWeight) {
        assertGreaterOrEqual<float64>("l2RegularizationWeight", l2RegularizationWeight, 0);
    }

    std::unique_ptr<ILabelWiseRuleEvaluation<DenseLabelWiseStatisticVector>> RegularizedLabelWiseRuleEvaluationFactory::createDense(
            const CompleteIndexVector& indexVector) const {
        return std::make_unique<DenseRegularizedLabelWiseRuleEvaluation<CompleteIndexVector>>(indexVector,
                                                                                              l2RegularizationWeight_);
    }

    std::unique_ptr<ILabelWiseRuleEvaluation<DenseLabelWiseStatisticVector>> RegularizedLabelWiseRuleEvaluationFactory::createDense(
            const PartialIndexVector& indexVector) const {
        return std::make_unique<DenseRegularizedLabelWiseRuleEvaluation<PartialIndexVector>>(indexVector,
                                                                                             l2RegularizationWeight_);
    }

}
